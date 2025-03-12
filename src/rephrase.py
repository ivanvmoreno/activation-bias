import argparse
import asyncio
import sys
import time
import json
from pathlib import Path

import litellm
from litellm.exceptions import RateLimitError, ServiceUnavailableError
import pandas as pd
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
    "larger_than",
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
    "smaller_than",
    "sp_en_trans_conj",
    "sp_en_trans",
    "sp_en_trans_de",
    "sp_en_trans_disj",
]

DATASETS_BASE_DIR = Path("./data/original")
OUTPUT_DIR = Path("./data/rephrased")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Combined JSONL output path
BATCH_JSONL_FILE = Path("./data/batch_items.jsonl")

# System/user prompts paths
with open("./src/prompts/sys_prompt.txt", "r", encoding="utf-8") as f:
    SYS_PROMPT = f.read()
with open("./src/prompts/user_prompt.txt", "r", encoding="utf-8") as f:
    USER_PROMPT = f.read()

# Few-shot examples CSV with columns: [dataset, original, rephrased]
few_shot_df = pd.read_csv("./data/few_shot_examples.csv")


class RequestRateLimiter:
    """
    A rate limiter that tracks and limits requests per minute in a rolling window.
    """

    def __init__(self, requests_per_minute: int):
        """
        :param requests_per_minute: Maximum number of requests allowed per minute.
        """
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire permission to make a request, waiting if necessary.
        """
        while True:
            async with self.lock:
                now = time.monotonic()
                # Remove timestamps older than 60 seconds
                window_start = now - 60
                self.request_timestamps = [
                    ts for ts in self.request_timestamps if ts > window_start
                ]

                if len(self.request_timestamps) < self.requests_per_minute:
                    self.request_timestamps.append(now)
                    return

            # If we are above the limit, compute how long to wait
            async with self.lock:
                oldest = self.request_timestamps[0]
            wait_time = 60 - (time.monotonic() - oldest)
            # Sleep until the oldest request in the window is outside the 60s window
            await asyncio.sleep(max(0.1, wait_time))


def get_few_shot_for_dataset(dataset_name: str, df_few_shot: pd.DataFrame) -> str:
    """
    Build a few-shot prompt snippet for the given dataset.
    """
    subset = df_few_shot[df_few_shot["dataset"] == dataset_name]
    examples = []
    for _, row in subset.iterrows():
        examples.append(f"Original: {row['original']}\nRephrased: {row['rephrased']}")
    return "\n\n".join(examples)


async def rephrase_statement(
    statement: str,
    few_shot_prompt: str,
    sys_prompt: str,
    user_prompt_template: str,
    model: str,
    api_key: str,
    rate_limiter: RequestRateLimiter,
    request_timeout: float = 30.0,
) -> str:
    """
    Sends one request to the OpenAI-like completion endpoint via litellm,
    rate-limited, and returns the rephrased statement.
    """
    # Wait for rate limiter permission
    await rate_limiter.acquire()

    # Prepare user content
    user_content = user_prompt_template.format(
        original_statement=statement, examples=few_shot_prompt
    )

    # Attempt the request
    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                api_key=api_key,
            ),
            timeout=request_timeout,
        )
        return response.choices[0].message.content.strip()

    except asyncio.TimeoutError:
        raise ServiceUnavailableError("Request timed out")


def build_combined_jsonl(datasets, base_dir, output_jsonl):
    """
    Reads each dataset's CSV, builds a combined JSONL of rows:
        {"dataset": <str>, "index": <int>, "statement": <str>}
    """
    items = []
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for dataset_name in datasets:
            csv_path = base_dir / f"{dataset_name}.csv"
            if not csv_path.exists():
                print(f"Skipping {csv_path} (not found)", file=sys.stderr)
                continue

            df = pd.read_csv(csv_path)
            if "statement" not in df.columns:
                print(f"Skipping {dataset_name} because 'statement' column not found.")
                continue

            for idx, row in df.iterrows():
                record = {
                    "dataset": dataset_name,
                    "index": idx,
                    "statement": row["statement"],
                }
                f.write(json.dumps(record) + "\n")
                items.append(record)

    return items


async def main():
    parser = argparse.ArgumentParser(
        description="Batch rephrasing across multiple datasets."
    )
    parser.add_argument("--model", required=True, help="Model to use (LiteLLM format)")
    parser.add_argument("--api-key", required=True, help="Provider API key")
    parser.add_argument(
        "--rpm", type=int, default=600, help="Requests per minute limit"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent worker tasks",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--rebuild-jsonl",
        action="store_true",
        help="Rebuild the combined JSONL file from CSVs",
    )
    args = parser.parse_args()

    model = args.model
    api_key = args.api_key
    rpm = args.rpm
    num_workers = args.workers
    request_timeout = args.timeout

    # Optionally rebuild the JSONL
    if args.rebuild_jsonl or not BATCH_JSONL_FILE.exists():
        print("Building combined JSONL file...")
        build_combined_jsonl(DATASETS, DATASETS_BASE_DIR, BATCH_JSONL_FILE)

    # Load all items from JSONL into memory
    all_items = []
    with open(BATCH_JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            all_items.append(json.loads(line.strip()))

    # Prepare a dictionary of {dataset_name: few_shot_string}
    dataset_few_shot_map = {
        ds: get_few_shot_for_dataset(ds, few_shot_df) for ds in DATASETS
    }

    # Store results in memory to then write them to CSV
    # Key by (dataset, index_in_csv), value = rephrased string
    results_map = {}

    # Unified queue of items to process
    # "priority=0" for initial attempt
    queue = asyncio.PriorityQueue()
    for i, item in enumerate(all_items):
        queue.put_nowait((0, i, item))

    rate_limiter = RequestRateLimiter(requests_per_minute=rpm)

    total_items = len(all_items)
    done_count = 0
    done_lock = asyncio.Lock()
    progress_event = asyncio.Event()

    async def worker():
        nonlocal done_count
        while True:
            try:
                priority, index, task_item = await queue.get()
            except asyncio.CancelledError:
                return

            dataset_name = task_item["dataset"]
            idx = task_item["index"]
            statement = task_item["statement"]

            try:
                few_shot_str = dataset_few_shot_map.get(dataset_name, "")
                rephrased = await rephrase_statement(
                    statement=statement,
                    few_shot_prompt=few_shot_str,
                    sys_prompt=SYS_PROMPT,
                    user_prompt_template=USER_PROMPT,
                    model=model,
                    api_key=api_key,
                    rate_limiter=rate_limiter,
                    request_timeout=request_timeout,
                )
                results_map[(dataset_name, idx)] = rephrased

                async with done_lock:
                    done_count += 1
                queue.task_done()
                progress_event.set()

            except (RateLimitError, ServiceUnavailableError):
                # Re-queue with higher priority
                queue.task_done()
                queue.put_nowait((priority - 1, index, task_item))
                # Slight backoff
                await asyncio.sleep(0.5)

            except Exception as e:
                # Unexpected error, re-queue at same priority
                print(f"Error on dataset={dataset_name}, idx={idx}: {e}")
                queue.task_done()
                queue.put_nowait((priority, index, task_item))
                await asyncio.sleep(1.0)

    async def progress_tracker():
        nonlocal done_count
        with tqdm(total=total_items, desc="Rephrasing") as pbar:
            last_done = 0
            while True:
                try:
                    await asyncio.wait_for(progress_event.wait(), timeout=1.0)
                    progress_event.clear()
                except asyncio.TimeoutError:
                    pass

                async with done_lock:
                    current_done = done_count
                if current_done > last_done:
                    pbar.update(current_done - last_done)
                    last_done = current_done

                if current_done >= total_items:
                    break

    # Start workers
    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
    progress_task = asyncio.create_task(progress_tracker())

    # Wait for the queue to be fully processed
    await queue.join()

    # Close workers
    for w in workers:
        w.cancel()

    await asyncio.gather(*workers, return_exceptions=True)
    await progress_task

    # Reconstruct output CSVs
    for dataset_name in DATASETS:
        csv_path = DATASETS_BASE_DIR / f"{dataset_name}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if "statement" not in df.columns:
            continue

        rephrased_col = []
        for idx in range(len(df)):
            rephrased = results_map.get((dataset_name, idx), "")
            rephrased_col.append(rephrased)

        df["statement_rephrased"] = rephrased_col

        # Reorder columns
        desired_order = ["statement", "label", "statement_rephrased"]
        # Keep original columns
        other_cols = [c for c in df.columns if c not in desired_order]

        # Replace existing columns in original dataset
        final_order = [c for c in desired_order if c in df.columns] + other_cols

        df = df[final_order]

        out_file = OUTPUT_DIR / f"{dataset_name}_rephrased.csv"
        df.to_csv(out_file, index=False)
        print(f"Saved rephrased output: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
