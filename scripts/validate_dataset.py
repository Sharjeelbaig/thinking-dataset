#!/usr/bin/env python3
"""Validate NeuroThinker CoT v2 dataset."""
import argparse, json
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {
    "id", "split", "category", "subcategory",
    "difficulty", "prompt", "thinking_cot", "answer", "source",
}

EXPECTED_COUNTS = {
    "logical_reasoning": 300,
    "multi_step_math": 250,
    "basic_arithmetic": 150,
    "causal_reasoning": 200,
    "pattern_recognition": 150,
    "coding_reasoning": 200,
    "common_sense": 200,
    "general_knowledge": 200,
    "conversational": 150,
    "problem_decomposition": 200,
}

TOTAL = 2000

def main():
    parser = argparse.ArgumentParser(description="Validate NeuroThinker dataset")
    parser.add_argument("--input", default="data/train.jsonl")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                raise ValueError(f"Row {i} missing fields: {sorted(missing)}")
            # Validate CoT quality: must be multi-step
            if len(row["thinking_cot"]) < 50:
                print(f"  ⚠ Row {i}: Short CoT ({len(row['thinking_cot'])} chars)")
            # Validate answer not empty
            if not row["answer"].strip():
                raise ValueError(f"Row {i}: Empty answer")
            rows.append(row)

    if len(rows) != TOTAL:
        raise ValueError(f"Expected {TOTAL} rows, got {len(rows)}")

    ids = [r["id"] for r in rows]
    if sorted(ids) != list(range(1, TOTAL + 1)):
        raise ValueError("IDs must be unique and contiguous 1..2000")

    cat_counts = Counter(r["category"] for r in rows)
    if dict(cat_counts) != EXPECTED_COUNTS:
        raise ValueError(f"Category mismatch:\n  Got: {dict(cat_counts)}\n  Expected: {EXPECTED_COUNTS}")

    print("✅ Validation OK")
    print(f"  Total rows: {len(rows)}")
    print(f"  Categories: {len(cat_counts)}")
    for cat in sorted(EXPECTED_COUNTS):
        print(f"    {cat}: {cat_counts[cat]}")

    # Quality stats
    avg_cot_len = sum(len(r["thinking_cot"]) for r in rows) / len(rows)
    avg_ans_len = sum(len(r["answer"]) for r in rows) / len(rows)
    print(f"  Avg CoT length: {avg_cot_len:.0f} chars")
    print(f"  Avg answer length: {avg_ans_len:.0f} chars")


if __name__ == "__main__":
    main()
