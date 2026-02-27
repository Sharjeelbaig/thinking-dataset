# Hugging Face Publishing Guide

This guide documents a real-world workflow to publish this dataset to Hugging Face Hub.

## 1) Prerequisites

- Python 3.9+
- Hugging Face CLI installed
- A Hugging Face account with an access token

Install/upgrade CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

Authenticate:

```bash
hf auth login
hf auth whoami
```

## 2) Generate and validate the dataset

From this repository root:

```bash
python3 scripts/generate_dataset.py
python3 scripts/validate_dataset.py
```

Expected result:

- `data/train.jsonl` with 1,000 rows
- `data/metadata.json` with schema and count summary

## 3) Create a dataset repository on HF

Choose your dataset ID (example below):

```bash
hf repo create Sharjeelbaig/thinking-cot-1k --repo-type dataset --exist-ok
```

## 4) Upload files with a clean commit message

```bash
hf upload Sharjeelbaig/thinking-cot-1k . . \
  --repo-type dataset \
  --exclude ".git/*" "__pycache__/*" "*.pyc" \
  --commit-message "Add Thinking CoT 1K dataset with README and generation scripts"
```

## 5) Verify the published dataset

- Open the dataset page on Hub and check:
  - Dataset card rendering (`README.md`)
  - File presence (`data/train.jsonl`, `data/metadata.json`)
  - Preview table integrity
- Validate loading from Python:

```python
from datasets import load_dataset
ds = load_dataset("Sharjeelbaig/thinking-cot-1k", split="train")
print(len(ds))
```

## 6) Recommended update workflow

When making changes:

```bash
python3 scripts/generate_dataset.py --seed 42
python3 scripts/validate_dataset.py
hf upload Sharjeelbaig/thinking-cot-1k . . --repo-type dataset
```

Use clear commit messages for each upload batch so downstream users can track data revisions.

## 7) Suggested versioning strategy

- Keep `main` as the latest stable dataset
- Tag milestone snapshots (`v1.0`, `v1.1`, etc.)
- Document schema changes in the dataset card changelog section
