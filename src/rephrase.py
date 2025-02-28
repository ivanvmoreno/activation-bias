import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import openai
import pandas as pd

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


async def generate_description(
    client: openai.AsyncOpenAI, query: str, model: str
) -> str:
    """Generate a description using OpenAI's Chat API with proper error handling and retries"""
    for attempt in range(RETRY_LIMIT):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI that converts audience segment queries into plain English descriptions.",
                    },
                    {
                        "role": "user",
                        "content": f"{PROMPT}{query}\nDescription:",
                    },
                ],
                max_tokens=150,
                temperature=0.7,
            )
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
    """Process rows in batches with shared OpenAI client instance"""
    client = openai.AsyncOpenAI(api_key=openai.api_key)
    rows = df.to_dict(orient="records")
    descriptions = []

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        tasks = [generate_description(client, r["statement"], MODEL) for r in batch]
        batch_descriptions = await asyncio.gather(*tasks)
        descriptions.extend(batch_descriptions)
        await asyncio.sleep(1)  # Small delay to prevent rate limiting

    for row, desc in zip(rows, descriptions):
        row["statement_rephrased"] = desc
    return pd.DataFrame(rows)


async def main(input_file: Path, output_file: Path, api_key: str):
    """Main processing workflow"""
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError(
            "Missing OpenAI API key. Provide via --api-key or environment variables."
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
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Path to output CSV file",
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.api_key))
