import re
import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

SYS_PROMPT = "You are an AI that rephrases short statements into longer, more detailed descriptions"
PROMPT = """\
Assume the following statement is correct. Rephrase this into a much longer statement, without varying its meaning. Only output a single, longer, statement.

Examples:
Original: Fifty-one is larger than fifty-two.
Rephrased: It is a fact that the numerical value of fifty-one exceeds that of fifty-two, meaning that when comparing these two numbers, fifty-one holds a greater magnitude than fifty-two.
Original: Eighty-three is larger than fifty-three.
Rephrased: If one were to arrange eighty-three and fifty-three in descending order, eighty-three would appear before fifty-three, as the latter is undeniably smaller in value.

Original: {original_statement}
Rephrased:\
"""

MODEL = "gpt-4o-mini"
BATCH_SIZE = 5  # Limit concurrent requests to prevent rate limits
RETRY_LIMIT = 3  # Number of retries on failure


class TokenBucket:
    """
    A simple token bucket implementation to enforce a maximum token consumption rate.
    The bucket refills continuously at the specified rate (tokens per second).
    """

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, amount: int):
        """
        Consume a specified amount of tokens. If insufficient tokens are available,
        wait until enough tokens have been refilled.
        """
        async with self.lock:
            self._refill()
            if self.tokens >= amount:
                self.tokens -= amount
                return
            else:
                needed = amount - self.tokens
        wait_time = needed / self.refill_rate
        await asyncio.sleep(wait_time)
        async with self.lock:
            self._refill()
            self.tokens = max(0, self.tokens - amount)

    async def refund(self, amount: int):
        """
        Refund unused tokens back to the bucket (without exceeding capacity).
        """
        async with self.lock:
            self.tokens = min(self.capacity, self.tokens + amount)

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


async def rephrase(
    client: openai.AsyncOpenAI, query: str, model: str, token_bucket: TokenBucket
) -> str:
    """
    Generate a description using OpenAI's Chat API with proper error handling, retries,
    and token bucket rate-limiting. It reserves an estimated number of tokens before the call
    and refunds any unused tokens after the response.
    """
    estimated_tokens = 200  # Estimated tokens consumption for this request
    await token_bucket.consume(estimated_tokens)
    for attempt in range(RETRY_LIMIT):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYS_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": PROMPT.format(original_statement=query),
                    },
                ],
            )
            try:
                actual_tokens = response.usage.total_tokens
            except AttributeError:
                actual_tokens = estimated_tokens
            refund_amount = estimated_tokens - actual_tokens
            if refund_amount > 0:
                await token_bucket.refund(refund_amount)
            return response.choices[0].message.content.strip()
        except openai.OpenAIError as e:
            if "rate_limit_exceeded" in str(e).lower():
                wait_time = (2**attempt) * 0.5  # Exponential backoff
                print(
                    f"Rate limit reached. Retrying in {wait_time:.2f} seconds...",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait_time)
            else:
                print(f"Error processing query: {e}", file=sys.stderr)
                return "ERROR"
    return "ERROR (Max retries exceeded)"


async def process_batch(df: pd.DataFrame, batch_size=BATCH_SIZE) -> pd.DataFrame:
    """
    Process CSV rows in batches with a shared OpenAI client instance and token bucket.
    Each row's 'statement' field is processed to generate a rephrased statement.
    """
    client = openai.AsyncOpenAI(api_key=openai.api_key)
    rows = df.to_dict(orient="records")
    descriptions = []
    total_rows = len(rows)
    pbar = tqdm(total=total_rows, desc="Processing queries", unit="query")

    # Create a token bucket: 30,000 tokens capacity, refilling at 500 tokens per second.
    token_bucket = TokenBucket(capacity=30000, refill_rate=500)

    for i in range(0, total_rows, batch_size):
        batch = rows[i : i + batch_size]
        tasks = [rephrase(client, r["statement"], MODEL, token_bucket) for r in batch]
        batch_descriptions = await asyncio.gather(*tasks)
        descriptions.extend(batch_descriptions)
        pbar.update(len(batch))
    pbar.close()

    for row, desc in zip(rows, descriptions):
        row["statement_rephrased"] = desc
    return pd.DataFrame(rows)


async def main(input_file: Path, output_file: Path, api_key: str):
    """
    Main processing workflow: load the CSV, process rows asynchronously using the shared token bucket,
    and write the outputs to a CSV file.
    """
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError(
            "Missing OpenAI API key. Provide via --api-key or set the OPENAI_API_KEY environment variable."
        )

    df = pd.read_csv(input_file)
    result_df = await process_batch(df)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"Completed processing {len(result_df)} rows and saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, type=Path, help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Path to output CSV file"
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.api_key))
