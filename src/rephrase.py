import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import openai
import pandas as pd
from openai import RateLimitError
from tqdm import tqdm

DATASETS = [
    "animal_class_conj",
    "animal_class",
    "animal_class_de",
    "animal_class_disj",
    "cities_conj",
    "cities",
    "cities_de",
    "cities_disj",
    "common_claim_true_false",
    "counterfact_true_false",
    "element_symb_conj",
    "element_symb",
    "element_symb_de",
    "element_symb_disj",
    "facts_conj",
    "facts",
    "facts_de",
    "facts_disj",
    "inventors_conj",
    "inventors",
    "inventors_de",
    "inventors_disj",
    # "larger_than",
    "neg_animal_class",
    "neg_animal_class_de",
    "neg_cities",
    "neg_cities_de",
    "neg_element_symb",
    "neg_element_symb_de",
    "neg_facts",
    "neg_facts_de",
    "neg_inventors",
    "neg_inventors_de",
    "neg_sp_en_trans",
    "neg_sp_en_trans_de",
    # "smaller_than",
    "sp_en_trans_conj",
    "sp_en_trans",
    "sp_en_trans_de",
    "sp_en_trans_disj",
]

MODEL = "gpt-4o-mini"
DATASETS_BASE_DIR = Path("./data/original")
OUTPUT_DIR = Path("./data/rephrased")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load system/user prompts
with open("./src/prompts/sys_prompt.txt", "r", encoding="utf-8") as f:
    SYS_PROMPT = f.read()
with open("./src/prompts/user_prompt.txt", "r", encoding="utf-8") as f:
    USER_PROMPT = f.read()

# Few-shot examples, a CSV with columns: [dataset, original, rephrased]
few_shot_df = pd.read_csv("./data/few_shot_examples.csv")

# Concurrency + rate-limit settings
BATCH_SIZE = 5  # Number of concurrent workers
ESTIMATED_TOKENS = 200  # Adjust if you have a better estimate
TOKEN_BUCKET_CAPACITY = 200000
TOKEN_BUCKET_REFILL_RATE = 500


class TokenBucket:
    """
    A simple token bucket to enforce a maximum token consumption rate.
    Refill at `refill_rate` tokens/second, up to `capacity`.
    """

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    async def consume(self, amount: int):
        """Block until we have enough tokens for `amount`."""
        async with self.lock:
            self._refill()
            if self.tokens >= amount:
                self.tokens -= amount
                return
            needed = amount - self.tokens
        # Wait outside the lock
        await asyncio.sleep(needed / self.refill_rate)
        async with self.lock:
            self._refill()
            self.tokens = max(0, self.tokens - amount)

    async def refund(self, amount: int):
        async with self.lock:
            self.tokens = min(self.capacity, self.tokens + amount)


def get_few_shot_examples(dataset_name: str, few_shot_df: pd.DataFrame) -> str:
    """
    Return a string with each example:
    Original: ...
    Rephrased: ...
    """
    df_filtered = few_shot_df[few_shot_df["dataset"] == dataset_name]
    examples = []
    for _, row in df_filtered.iterrows():
        examples.append(f"Original: {row['original']}\nRephrased: {row['rephrased']}")
    return "\n\n".join(examples)


async def rephrase_statement(
    statement: str,
    few_shot: str,
    token_bucket: TokenBucket,
    client: "openai.AsyncOpenAI",
) -> str:
    """
    Rephrase one statement using openai.ChatCompletion.
    Includes token bucket consumption and partial refund logic.
    """
    user_content = USER_PROMPT.format(original_statement=statement, examples=few_shot)

    # Attempt to consume our estimated tokens first
    await token_bucket.consume(ESTIMATED_TOKENS)

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1000,
    )
    # Refund any tokens we overestimated
    try:
        actual_tokens = response.usage.total_tokens
    except AttributeError:
        actual_tokens = ESTIMATED_TOKENS

    refund_amount = ESTIMATED_TOKENS - actual_tokens
    if refund_amount > 0:
        await token_bucket.refund(refund_amount)

    return response.choices[0].message.content.strip()


async def process_dataset(dataset: str, client: "openai.AsyncOpenAI"):
    input_file = DATASETS_BASE_DIR / f"{dataset}.csv"
    output_file = OUTPUT_DIR / f"{dataset}_rephrased.csv"

    if not input_file.exists():
        print(
            f"Input file {input_file} not found. Skipping {dataset}.", file=sys.stderr
        )
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # Get few-shot examples for this dataset
    few_shot = get_few_shot_examples(dataset, few_shot_df)

    # Prepare the token bucket
    token_bucket = TokenBucket(
        capacity=TOKEN_BUCKET_CAPACITY, refill_rate=TOKEN_BUCKET_REFILL_RATE
    )

    total_items = len(df)
    results = [""] * total_items

    # We will track how many are done so we know when to stop
    done_count = 0
    done_count_lock = asyncio.Lock()

    # Create a queue of (index, statement) for all items
    queue = asyncio.Queue()
    for i, row in df.iterrows():
        queue.put_nowait((i, row["statement"]))

    async def worker():
        nonlocal done_count
        while True:
            try:
                idx, statement = queue.get_nowait()
            except asyncio.QueueEmpty:
                # If queue is empty, check if we're done
                # If not done, yield to let other tasks do work or refill
                await asyncio.sleep(0.1)
                continue

            try:
                rephrased = await rephrase_statement(
                    statement, few_shot, token_bucket, client
                )
                results[idx] = rephrased
                # Mark done
                async with done_count_lock:
                    done_count += 1
                queue.task_done()
            except RateLimitError:
                # Global backoff, re-queue
                queue.task_done()
                queue.put_nowait((idx, statement))
                await asyncio.sleep(3.0)  # or dynamic backoff
            except openai.OpenAIError:
                # Re-queue indefinitely
                queue.task_done()
                queue.put_nowait((idx, statement))
                # You might add a small delay here to avoid tight loop
                await asyncio.sleep(1.0)

    # Spawn workers
    workers = [asyncio.create_task(worker()) for _ in range(BATCH_SIZE)]

    # Track progress in a loop until we've completed all items
    with tqdm(total=total_items, desc=dataset) as pbar:
        last_done = 0
        while True:
            async with done_count_lock:
                current_done = done_count
            new_done = current_done - last_done
            if new_done > 0:
                pbar.update(new_done)
                last_done = current_done

            if current_done >= total_items:
                # All items done
                break

            await asyncio.sleep(0.5)

    # All items are done
    # Cancel all workers
    for w in workers:
        w.cancel()

    # Insert rephrased statements
    df["statement_rephrased"] = results
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rephrased statements to {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Real-time rephrasing for multiple datasets (no retry limit)."
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--dataset", help="Specific dataset to process (otherwise all)")

    args = parser.parse_args()

    openai.api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError(
            "Missing OpenAI API key. Provide via --api-key or set OPENAI_API_KEY."
        )

    client = openai.AsyncOpenAI(api_key=openai.api_key)

    if args.dataset:
        # Process a specific dataset
        await process_dataset(args.dataset, client)
    else:
        # Process all datasets
        tasks = [asyncio.create_task(process_dataset(ds, client)) for ds in DATASETS]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
