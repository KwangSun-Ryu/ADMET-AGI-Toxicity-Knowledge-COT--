# main.py
import os
import sys
import shlex
import time
import argparse
import random
from datetime import datetime

from utils import load_json, save_list_to_json, FileLogger
from evaluate import run_batched
from scoring import evaluate_accuracy_only

def main():
    parser = argparse.ArgumentParser(description="Batch inference runner (requests) for custom FastAPI vLLM server")
    parser.add_argument("--input", default=os.getenv("INPUT_JSON", "/path/to/input.json"))
    parser.add_argument("--output", default=os.getenv("OUTPUT_JSON", "/path/to/output.json"))
    parser.add_argument("--log_dir", default=os.getenv("LOG_DIR", "./logs"))
    parser.add_argument("--retries", type=int, default=int(os.getenv("RETRIES", "2")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "32")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("REQ_TIMEOUT", "300")))
    args = parser.parse_args()

    base_url = os.getenv("INFER_BASE_URL", "http://localhost:30001")
    infer_batch_path = os.getenv("INFER_BATCH_PATH", "/generate_batch")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"run_{run_id}.txt")
    logger = FileLogger(log_path)

    cmdline = " ".join(shlex.quote(x) for x in sys.argv)
    logger.log(f"Command: {cmdline}")
    logger.log(f"Input: {args.input}")
    logger.log(f"Output: {args.output}")
    logger.log(f"Base URL: {base_url}")
    logger.log(f"Batch path: {infer_batch_path}")
    logger.log(f"Batch size: {args.batch_size}, Retries: {args.retries}, Timeout: {args.timeout}")

    random.seed(42)
    t0 = time.time()

    data = load_json(args.input)
    logger.log(f"Loaded {len(data)} records from input.")

    results = run_batched(
        data=data,
        base_url=base_url,
        path=infer_batch_path,
        batch_size=args.batch_size,
        logger=logger,
        retries=args.retries,
        request_timeout=args.timeout,
    )

    for d, out in zip(data, results):
        d["model_response"] = out
        d["LABEL"] = 1 if d["compound_info"]["toxicity"]["activity"] == "Active" else 0

    acc = evaluate_accuracy_only(
        data, answer_key="model_response", label_key="LABEL", count_missing_as_wrong=False
    )
    logger.log(f"Accuracy (scored only): {acc:.4f}")

    save_list_to_json(data, args.output)
    dt = time.time() - t0
    logger.log(f"Saved results to {args.output}")
    logger.log(f"Elapsed: {dt:.2f} sec")
    logger.log("Done.")

if __name__ == "__main__":
    main()
