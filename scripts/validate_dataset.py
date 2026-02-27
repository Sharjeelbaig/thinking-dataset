#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {
    "id",
    "split",
    "category",
    "subcategory",
    "difficulty",
    "prompt",
    "thinking_cot",
    "answer",
    "source",
}

EXPECTED_COUNTS = {
    "basic_arithmetic": 200,
    "advanced_math": 200,
    "general_chat": 400,
    "coding_tasks": 200,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Thinking CoT dataset")
    parser.add_argument("--input", default="data/train.jsonl", help="Path to JSONL file")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                raise ValueError(f"Row {i} is missing fields: {sorted(missing)}")
            rows.append(row)

    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 rows, found {len(rows)}")

    id_values = [row["id"] for row in rows]
    if sorted(id_values) != list(range(1, 1001)):
        raise ValueError("IDs must be unique and contiguous from 1 to 1000")

    category_counts = Counter(row["category"] for row in rows)
    if dict(category_counts) != EXPECTED_COUNTS:
        raise ValueError(f"Category counts mismatch: {dict(category_counts)} != {EXPECTED_COUNTS}")

    split_values = {row["split"] for row in rows}
    if split_values != {"train"}:
        raise ValueError(f"Unexpected split values: {sorted(split_values)}")

    print("Validation OK")
    print(f"Total rows: {len(rows)}")
    print("Category counts:")
    for k, v in EXPECTED_COUNTS.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
