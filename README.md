---
language:
- en
license: cc-by-4.0
task_categories:
- text-generation
- question-answering
- other
tags:
- reasoning
- chain-of-thought
- arithmetic
- mathematics
- coding
- synthetic
size_categories:
- 1K<n<10K
pretty_name: Thinking CoT 1K
---

# Thinking CoT 1K

A production-style synthetic dataset for reasoning-focused LLM/SLM training.

This dataset provides 1,000 instruction-response samples with explicit `thinking_cot` traces for multi-domain supervision:

- Basic arithmetic reasoning (200)
- Advanced math reasoning (200)
- General chat + explanation behavior (400)
- Coding and snippet generation tasks (200)

## Why this dataset

Reasoning-focused fine-tuning often needs a clean, structured mix of:

- Deterministic numeric tasks (for answer correctness)
- Multi-step analytical tasks (for reasoning depth)
- Conversational tasks (for instruction following and tone)
- Code generation tasks (for practical developer workflows)

Thinking CoT 1K is designed as a bootstrap dataset to help teams prototype reasoning-capable assistants quickly.

## Dataset structure

Single split:

- `train`: 1,000 rows

Schema per row:

- `id` (int): contiguous row id, `1..1000`
- `split` (str): `train`
- `category` (str): `basic_arithmetic`, `advanced_math`, `general_chat`, `coding_tasks`
- `subcategory` (str): more specific task type within each category
- `difficulty` (str): `easy`, `medium`, `hard`
- `prompt` (str): user-style task instruction
- `thinking_cot` (str): short reasoning trace to guide chain-of-thought style learning
- `answer` (str): expected output (plain text, numeric result, or code snippet)
- `source` (str): `synthetic`

## Category distribution

- `basic_arithmetic`: 200
- `advanced_math`: 200
- `general_chat`: 400
- `coding_tasks`: 200

Total: **1,000 rows**

## Advanced math coverage

Advanced section includes a realistic range of topics:

- `log10`, `ln`
- `sin`, `cos`, `cosecant` (degree-based)
- 2x2 matrix multiplication
- 3x3 determinant computation
- Tensor-focused shape reasoning (`permute`, `matmul`, `broadcast`, `reshape`)

## Quickstart (Python)

```python
from datasets import load_dataset

# Replace with your final dataset ID if you fork or rename it
ds = load_dataset("Sharjeelbaig/thinking-cot-1k", split="train")

print(ds[0])
print(ds.features)
```

## Recommended training format

For supervised fine-tuning, a robust format is:

```text
User: {prompt}
Assistant (thinking): {thinking_cot}
Assistant (final): {answer}
```

Or hide `thinking_cot` at inference time and train your model to internally reason before emitting `answer`.

## Reproducibility

Dataset is generated from source scripts in this repository:

- `scripts/generate_dataset.py`
- `scripts/validate_dataset.py`

Generate and validate locally:

```bash
python3 scripts/generate_dataset.py
python3 scripts/validate_dataset.py
```

## Limitations

- Data is synthetic and templated; it should be mixed with real-world human data for strong production performance.
- `thinking_cot` traces are concise by design and do not cover every possible reasoning strategy.
- Advanced mathematics here is broad but not exhaustive for formal theorem-heavy tasks.

## License

This dataset is released under **CC BY 4.0**.
