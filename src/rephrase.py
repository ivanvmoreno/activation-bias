import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import openai
import pandas as pd
from openai import RateLimitError
from tqdm import tqdm

# ------------------------------------------------------------------------
# Global config – replace or set these as you see fit
# ------------------------------------------------------------------------

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

# Load system/user prompts from file. Adapt the paths to your environment.
with open("./src/prompts/sys_prompt.txt", "r", encoding="utf-8") as f:
    SYS_PROMPT = f.read()
with open("./src/prompts/user_prompt.txt", "r", encoding="utf-8") as f:
    USER_PROMPT = f.read()

# A CSV with columns: [dataset, original, rephrased], each row is a few-shot example.
few_shot_df = pd.read_csv("./data/few_shot_examples.csv")

# Concurrency + rate-limit settings
BATCH_SIZE = 5  # Number of concurrent requests
RETRY_LIMIT = 3  # Number of retries on error (esp. rate limit)
ESTIMATED_TOKENS = 200  # Adjust if you have a better estimate
TOKEN_BUCKET_CAPACITY = 30000
TOKEN_BUCKET_REFILL_RATE = 500


# ------------------------------------------------------------------------
# A simple Token Bucket for rate-limiting – from your first script
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# Helper for retrieving few-shot examples for a given dataset
# ------------------------------------------------------------------------
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


# ------------------------------------------------------------------------
# The core rephrase call – real-time calls, no batch job
# ------------------------------------------------------------------------
async def rephrase_statement(
    statement: str,
    few_shot: str,
    token_bucket: TokenBucket,
    client: "openai.AsyncOpenAI",
) -> str:
    """
    Rephrase one statement in real-time using openai.ChatCompletion.
    Includes token bucket consumption and basic retry logic for rate-limit.
    """
    # Construct the user content with few-shot examples and the new statement
    user_content = USER_PROMPT.format(original_statement=statement, examples=few_shot)

    # Attempt to consume our estimated tokens first
    await token_bucket.consume(ESTIMATED_TOKENS)

    for attempt in range(RETRY_LIMIT):
        try:
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

        except RateLimitError:
            # Rate limit error – exponential backoff
            wait_time = (2**attempt) * 0.5
            print(f"Rate limit error. Retrying in {wait_time:.2f}s...", file=sys.stderr)
            await asyncio.sleep(wait_time)
        except openai.OpenAIError as e:
            # Some other OpenAI error – optionally retry, or just break
            print(f"OpenAI error: {e}", file=sys.stderr)
            # You can break or continue for more tries. Here, we'll break:
            break

    return "ERROR (Max retries or failure)"


# ------------------------------------------------------------------------
# Process one dataset – read CSV, rephrase each statement concurrently, save CSV
# ------------------------------------------------------------------------
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

    # We'll process in concurrency-limited batches
    results = [None] * len(df)

    async def worker(idx: int, statement: str):
        resp = await rephrase_statement(statement, few_shot, token_bucket, client)
        results[idx] = resp

    tasks = []
    semaphore = asyncio.Semaphore(BATCH_SIZE)

    async def sem_task(idx: int, statement: str):
        async with semaphore:
            return await worker(idx, statement)

    # Create all tasks
    for i, row in df.iterrows():
        tasks.append(asyncio.create_task(sem_task(i, row["statement"])))

    # Show a progress bar while tasks run
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=dataset):
        await f

    # Insert rephrased statements
    df["statement_rephrased"] = results
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rephrased statements to {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Real-time rephrasing for multiple datasets"
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--dataset", help="Specific dataset to process (otherwise all)")

    args = parser.parse_args()

    openai.api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError(
            "Missing OpenAI API key. Provide via --api-key or set OPENAI_API_KEY."
        )

    # Use the async OpenAI client
    client = openai.AsyncOpenAI(api_key=openai.api_key)

    if args.dataset:
        # Process just one dataset
        await process_dataset(args.dataset, client)
    else:
        # Process all datasets concurrently
        tasks = [asyncio.create_task(process_dataset(ds, client)) for ds in DATASETS]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
