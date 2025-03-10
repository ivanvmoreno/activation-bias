import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import OpenAI
from openai import RateLimitError
import pandas as pd

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

with open("./src/prompts/sys_prompt.txt", "r", encoding="utf-8") as f:
    SYS_PROMPT = f.read()
with open("./src/prompts/user_prompt.txt", "r", encoding="utf-8") as f:
    USER_PROMPT = f.read()

few_shot_df = pd.read_csv("./data/few_shot_examples.csv")


def get_few_shot_examples(dataset_name: str, few_shot_df: pd.DataFrame) -> str:
    """
    Retrieve few-shot examples for the given dataset as a formatted string.
    """
    df_filtered = few_shot_df[few_shot_df["dataset"] == dataset_name]
    examples = []
    for _, row in df_filtered.iterrows():
        examples.append(f"Original: {row['original']}\nRephrased: {row['rephrased']}")
    return "\n\n".join(examples)


def create_batch_file(df: pd.DataFrame, batch_file: Path, few_shot: str):
    """
    Create a JSONL file containing batch requests.
    Each line is a JSON object that includes 'method', 'url', 'body',
    and a 'custom_id' for matching. The user prompt is formatted with both
    the original statement and few-shot examples.
    """
    with open(batch_file, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            user_content = USER_PROMPT.format(
                original_statement=row["statement"], examples=few_shot
            )
            request = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 1000,
                },
            }
            json.dump(request, f)
            f.write("\n")


async def upload_file(batch_file: Path, client: OpenAI) -> str:
    """
    Upload the JSONL batch file to OpenAI using the single client object.
    """
    with open(batch_file, "rb") as file_obj:
        uploaded_file = await asyncio.to_thread(
            client.files.create, file=file_obj, purpose="batch"
        )
    print(f"Uploaded file: {uploaded_file.id}")
    return uploaded_file.id


async def create_batch_job(
    file_id: str, client: OpenAI, description: str, completion_window="24h"
) -> object:
    """
    Create a batch job using the uploaded file via the single client object.
    Returns a BatchObject, whose attributes can be accessed (e.g. job.id).
    """
    job = await asyncio.to_thread(
        client.batches.create,
        endpoint="/v1/chat/completions",
        input_file_id=file_id,
        metadata={"description": description},
        completion_window=completion_window,
    )
    print(f"Created batch job with id: {job.id}")
    return job


async def poll_batch_job(
    job_id: str, client: OpenAI, poll_interval: int = 60
) -> object:
    """
    Poll the batch job status every `poll_interval` seconds until a terminal state is reached.
    Terminal statuses include: "completed", "failed", "expired", or "canceled".
    Returns the final BatchObject.
    """
    while True:
        job = await asyncio.to_thread(client.batches.retrieve, job_id)
        status = job.status
        print(f"Job {job_id} status: {status}")
        if status in ["completed", "failed", "expired", "canceled"]:
            return job
        await asyncio.sleep(poll_interval)


async def download_results(
    output_file_id: str, destination: Path, client: OpenAI
) -> None:
    """
    Download the output file (JSONL) containing batch results.
    """
    file_content = await asyncio.to_thread(
        lambda: client.files.content(output_file_id).read()
    )
    with open(destination, "wb") as f:
        f.write(file_content)
    print(f"Downloaded results file to {destination}")


async def process_batch_job(
    input_file: Path, output_file: Path, client: OpenAI, few_shot: str
):
    """
    Main workflow:
      1. Read the input CSV.
      2. Create a JSONL batch file from CSV rows.
      3. Upload the batch file and create a batch job.
      4. Poll the job until completion.
      5. Download and parse the results.
      6. Merge the rephrased statements back into the CSV and save.
    Each dataset CSV is assumed to have columns: statement, label.
    """
    if not client.api_key:
        raise ValueError(
            "Missing OpenAI API key. Provide via --api-key or set the OPENAI_API_KEY environment variable."
        )

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    batch_file = input_file.parent / "batch_requests.jsonl"
    create_batch_file(df, batch_file, few_shot)
    print(f"Created batch file: {batch_file}")

    file_id = await upload_file(batch_file, client)

    job = await create_batch_job(file_id, client, "Batch job to rephrase statements")
    job_id = job.id

    job = await poll_batch_job(job_id, client)
    status = job.status
    if status != "completed":
        print(f"Batch job ended with status: {status}", file=sys.stderr)
        return

    output_file_id = job.output_file_id
    if not output_file_id:
        print("No output file id found", file=sys.stderr)
        return

    results_file = input_file.parent / "batch_results.jsonl"
    await download_results(output_file_id, results_file, client)

    results = {}
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result.get("custom_id")
            try:
                rephrased = result["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                rephrased = "ERROR"
            results[custom_id] = rephrased

    df["statement_rephrased"] = df.index.astype(str).map(results)
    df.to_csv(output_file, index=False)
    print(f"Completed processing. Saved results to {output_file}")


async def run_for_dataset(dataset: str, client: OpenAI):
    """
    Enqueue a dataset's rephrasing job. If a rate limit error occurs during enqueuing,
    wait (for example, 60 seconds) before retrying.
    """
    input_file = DATASETS_BASE_DIR / f"{dataset}.csv"
    output_file = OUTPUT_DIR / f"{dataset}_rephrased.csv"
    if not input_file.exists():
        print(
            f"Input file {input_file} does not exist. Skipping dataset {dataset}.",
            file=sys.stderr,
        )
        return

    print(f"Processing dataset: {dataset}")
    few_shot = get_few_shot_examples(dataset, few_shot_df)
    while True:
        try:
            # Attempt to process the dataset (enqueue the batch job)
            await process_batch_job(input_file, output_file, client, few_shot)
            break  # Exit loop if successful
        except RateLimitError as e:
            print(
                f"Rate limit error while enqueuing {dataset}: {e}. Waiting for current batches to complete before retrying."
            )
            await asyncio.sleep(60)  # Wait 60 seconds before retrying
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}", file=sys.stderr)
            break


async def main():
    parser = argparse.ArgumentParser(
        description="Batch rephrase multiple datasets using the OpenAI Batch API"
    )
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument(
        "--dataset",
        help="Specific dataset name to process (if not provided, all datasets will be processed)",
    )
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY"))

    if args.dataset:
        await run_for_dataset(args.dataset, client)
    else:
        # Enqueue all datasets concurrently.
        tasks = [
            asyncio.create_task(run_for_dataset(dataset, client))
            for dataset in DATASETS
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
