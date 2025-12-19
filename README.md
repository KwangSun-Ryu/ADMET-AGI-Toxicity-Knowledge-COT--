# Toxicity Knowledge CoT Framework

This repository provides a batch inference and evaluation pipeline
designed to support toxicity knowledge Chain-of-Thought (CoT) evaluation.

The framework sends JSON-formatted samples to the server,
collects model responses containing structured reasoning,
and evaluates binary toxicity classification accuracy.

---

## Features

- Batch inference via `/generate_batch`
- Automatic fallback to single inference (`/generate`) on failure
- Retry with exponential backoff
- Ground-truth label generation from toxicity annotations
- Accuracy computation based on parsed model answers
- Full execution logging

---

## File Structure

```
.
├── main.py          # Entry point: end-to-end pipeline execution
├── evaluate.py      # Batch & single inference logic
├── scoring.py       # Model output parsing & accuracy calculation
├── utils.py         # JSON I/O, logging, prompt & CoT utilities
├── pipeline.sh      # Example execution script
└── README.md
```

---

## Input Data Format

Input must be a **JSON array**.
Each item should include toxicity knowledge and metadata as follows:

```json
[
  {
    "compound_info": {
      "toxicity": {
        "activity": "Active"
      }
    }
  }
]
```

### Ground Truth Label

- `"activity" == "Active"` → `LABEL = 1`
- otherwise → `LABEL = 0`

---

## Output Data Format

The output JSON preserves the input structure and adds:

```json
{
  "model_response": "...",
  "LABEL": 1
}
```

`model_response` may include **reasoning steps (CoT)** and a final answer.

---

## Model Output Parsing

The framework expects the model output to include a final answer tag:

```xml
<answer>독성</answer>
```

or

```xml
<answer>비독성</answer>
```

Parsing rules:

- `독성` → prediction = 1
- `비독성` → prediction = 0

Samples without a valid `<answer>` tag are excluded from scoring.
Accuracy is computed over **parsable samples only**.

---

## Environment Variables

| Variable             | Description              | Default                    |
| -------------------- | ------------------------ | -------------------------- |
| `INFER_BASE_URL`   | vLLM server base URL     | `http://localhost:30001` |
| `INFER_BATCH_PATH` | Batch inference endpoint | `/generate_batch`        |
| `INPUT_JSON`       | Input JSON path          | None                       |
| `OUTPUT_JSON`      | Output JSON path         | None                       |
| `BATCH_SIZE`       | Batch size               | `32`                     |
| `RETRIES`          | Retry count              | `2`                      |
| `REQ_TIMEOUT`      | Request timeout (sec)    | `300`                    |

---

## Execution

### Using pipeline.sh

```bash
bash pipeline.sh
```

### Direct execution

```bash
python main.py \
  --input input.json \
  --output output.json \
  --batch_size 32 \
  --log_dir ./logs
```

---

## Pipeline Flow

```
input.json
   → vLLM inference (with CoT reasoning)
   → model_response collection
   → LABEL generation
   → answer parsing
   → accuracy computation
   → output.json + logs
```

---

## Logs

- Logs are saved to:
  ```
  logs/run_YYYYMMDD_HHMMSS.txt
  ```
- Includes:
  - Execution arguments
  - Request/response logs
  - Accuracy summary
  - Runtime statistics

---

## Notes

- This repository **does not include the vLLM inference server**.
- Ensure the FastAPI-based vLLM server is running before execution.
- For large-scale experiments, adjust batch size and timeout settings.

---

## Summary

- Toxicity knowledge CoT evaluation framework
- FastAPI vLLM-based batch inference
- Robust retry & fallback handling
- Binary toxicity classification scoring
- Ready for large-scale offline evaluation
