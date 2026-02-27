#!/usr/bin/env python3
import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Row:
    id: int
    split: str
    category: str
    subcategory: str
    difficulty: str
    prompt: str
    thinking_cot: str
    answer: str
    source: str = "synthetic"

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "split": self.split,
            "category": self.category,
            "subcategory": self.subcategory,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "thinking_cot": self.thinking_cot,
            "answer": self.answer,
            "source": self.source,
        }


def fmt_num(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    if abs(value - int(value)) < 1e-10:
        return str(int(value))
    out = f"{value:.6f}".rstrip("0").rstrip(".")
    return out if out else "0"


def build_basic_arithmetic(start_id: int, count: int, rng: random.Random) -> List[Row]:
    rows: List[Row] = []
    ops = ["add", "subtract", "multiply", "divide", "modulus", "power", "average", "percentage"]

    for i in range(count):
        op = ops[i % len(ops)]
        row_id = start_id + i

        if op == "add":
            a, b = rng.randint(-500, 500), rng.randint(-500, 500)
            answer = a + b
            prompt = f"Calculate {a} + {b}."
            cot = (
                f"Recognize this as addition. Combine the two signed values: {a} plus {b}. "
                "Report the resulting sum."
            )
            difficulty = "easy"

        elif op == "subtract":
            a, b = rng.randint(-800, 800), rng.randint(-800, 800)
            answer = a - b
            prompt = f"What is {a} - {b}?"
            cot = (
                "Treat subtraction as adding the opposite. "
                f"Start from {a}, then move by {-b} units. Return the final value."
            )
            difficulty = "easy"

        elif op == "multiply":
            a, b = rng.randint(-80, 80), rng.randint(-80, 80)
            answer = a * b
            prompt = f"Compute the product: {a} * {b}."
            cot = (
                "Identify multiplication of two integers. "
                f"Apply sign rules, then multiply absolute values |{a}| and |{b}|."
            )
            difficulty = "easy"

        elif op == "divide":
            b = rng.randint(1, 40) * (-1 if rng.random() < 0.2 else 1)
            quotient = rng.randint(-60, 60)
            remainder = rng.randint(0, abs(b) - 1)
            a = quotient * b + remainder
            answer = round(a / b, 6)
            prompt = f"Divide {a} by {b}. Give the result up to 6 decimal places."
            cot = (
                "Set up the fraction numerator/denominator and perform signed division. "
                "If not an integer, express as decimal rounded to 6 places."
            )
            difficulty = "medium"

        elif op == "modulus":
            a = rng.randint(10, 500)
            b = rng.randint(2, 31)
            answer = a % b
            prompt = f"Find {a} mod {b}."
            cot = (
                f"Compute how many full groups of {b} fit into {a}. "
                "Return the leftover remainder."
            )
            difficulty = "easy"

        elif op == "power":
            base = rng.randint(-9, 9)
            exp = rng.randint(2, 5)
            answer = base ** exp
            prompt = f"Evaluate {base}^{exp}."
            cot = (
                f"Exponentiation means multiply {base} by itself {exp} times. "
                "Track sign based on whether the exponent is even or odd."
            )
            difficulty = "medium"

        elif op == "average":
            values = [rng.randint(-100, 200) for _ in range(3)]
            total = sum(values)
            answer = round(total / len(values), 6)
            prompt = f"Find the average of {values[0]}, {values[1]}, and {values[2]}."
            cot = (
                "For mean, sum all given values then divide by count (3). "
                "Return the decimal result if needed."
            )
            difficulty = "easy"

        else:  # percentage
            base = rng.randint(20, 2000)
            pct = rng.choice([5, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 65, 75])
            answer = round((pct / 100.0) * base, 6)
            prompt = f"What is {pct}% of {base}?"
            cot = (
                f"Convert {pct}% to decimal ({pct}/100), multiply by {base}, "
                "and simplify the result."
            )
            difficulty = "easy"

        rows.append(
            Row(
                id=row_id,
                split="train",
                category="basic_arithmetic",
                subcategory=op,
                difficulty=difficulty,
                prompt=prompt,
                thinking_cot=cot,
                answer=fmt_num(answer),
            )
        )

    return rows


def matrix_mul_2x2(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def matrix_det_3x3(m: List[List[int]]) -> int:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def tuple_str(values: Tuple[int, ...]) -> str:
    return "(" + ", ".join(str(v) for v in values) + ")"


def build_advanced_math(start_id: int, count: int, rng: random.Random) -> List[Row]:
    rows: List[Row] = []

    plan = [
        ("log10", 25),
        ("ln", 15),
        ("sin_deg", 25),
        ("cos_deg", 25),
        ("cosecant_deg", 10),
        ("matrix_mul_2x2", 40),
        ("matrix_det_3x3", 20),
        ("tensor_shape", 40),
    ]

    if sum(x for _, x in plan) != count:
        raise ValueError("Advanced math category plan does not match requested count.")

    row_id = start_id
    for subcategory, qty in plan:
        for _ in range(qty):
            difficulty = "medium"

            if subcategory == "log10":
                x = round(10 ** rng.uniform(0.3, 5.0), 6)
                answer = round(math.log10(x), 6)
                prompt = f"Compute log10({fmt_num(x)}). Round to 6 decimals."
                cot = (
                    "Use the base-10 logarithm definition and evaluate numerically. "
                    "Round the final value to 6 decimal places."
                )

            elif subcategory == "ln":
                x = round(math.exp(rng.uniform(-2.0, 3.0)), 6)
                answer = round(math.log(x), 6)
                prompt = f"Find ln({fmt_num(x)}). Round to 6 decimals."
                cot = (
                    "Apply the natural logarithm to the positive input. "
                    "Compute and round the result to 6 decimals."
                )

            elif subcategory == "sin_deg":
                angle = rng.choice(list(range(0, 361, 5)))
                answer = round(math.sin(math.radians(angle)), 6)
                prompt = f"What is sin({angle}°)? Round to 6 decimals."
                cot = (
                    "Convert degrees to radians, evaluate sine, and round to 6 decimals."
                )

            elif subcategory == "cos_deg":
                angle = rng.choice(list(range(0, 361, 5)))
                answer = round(math.cos(math.radians(angle)), 6)
                prompt = f"Evaluate cos({angle}°). Round to 6 decimals."
                cot = (
                    "Convert the angle from degrees to radians, evaluate cosine, then round."
                )

            elif subcategory == "cosecant_deg":
                angle = rng.choice([30, 45, 60, 120, 135, 150, 210, 225, 240, 300, 315, 330])
                answer = round(1.0 / math.sin(math.radians(angle)), 6)
                prompt = f"Compute csc({angle}°). Round to 6 decimals."
                cot = (
                    "Use csc(theta) = 1/sin(theta). Evaluate sine for the angle and invert it."
                )
                difficulty = "hard"

            elif subcategory == "matrix_mul_2x2":
                a = [[rng.randint(-6, 8), rng.randint(-6, 8)], [rng.randint(-6, 8), rng.randint(-6, 8)]]
                b = [[rng.randint(-6, 8), rng.randint(-6, 8)], [rng.randint(-6, 8), rng.randint(-6, 8)]]
                answer = matrix_mul_2x2(a, b)
                prompt = f"Multiply matrices A and B where A={a} and B={b}."
                cot = (
                    "For each cell in AB, compute row-by-column dot products. "
                    "Assemble the four computed values into a 2x2 matrix."
                )
                difficulty = "hard"

            elif subcategory == "matrix_det_3x3":
                m = [[rng.randint(-5, 6) for _ in range(3)] for _ in range(3)]
                answer = matrix_det_3x3(m)
                prompt = f"Find the determinant of the 3x3 matrix: {m}."
                cot = (
                    "Use cofactor expansion (or Sarrus rule) for the 3x3 matrix and simplify."
                )
                difficulty = "hard"

            else:  # tensor_shape
                flavor = rng.choice(["permute", "matmul", "broadcast_add", "reshape"])

                if flavor == "permute":
                    shape = (rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 6))
                    perm = list(range(4))
                    rng.shuffle(perm)
                    new_shape = tuple(shape[i] for i in perm)
                    prompt = (
                        f"A tensor has shape {tuple_str(shape)}. After permuting axes to {tuple_str(tuple(perm))}, "
                        "what is the new shape?"
                    )
                    answer = tuple_str(new_shape)
                    cot = (
                        "Permutation reorders dimensions by axis index. "
                        "Map each new axis to the original axis and write the resulting shape."
                    )

                elif flavor == "matmul":
                    a = (rng.randint(2, 8), rng.randint(2, 8))
                    b = (a[1], rng.randint(2, 8))
                    out = (a[0], b[1])
                    prompt = (
                        f"If matrix A has shape {tuple_str(a)} and matrix B has shape {tuple_str(b)}, "
                        "what is the shape of A @ B?"
                    )
                    answer = tuple_str(out)
                    cot = (
                        "Matrix multiplication keeps outer dimensions and contracts inner matched dimension. "
                        "Result shape is (rows of A, columns of B)."
                    )

                elif flavor == "broadcast_add":
                    a = (rng.randint(2, 6), 1, rng.randint(2, 7))
                    b = (1, rng.randint(2, 6), a[2])
                    out = (a[0], b[1], a[2])
                    prompt = (
                        f"Tensor X has shape {tuple_str(a)} and Y has shape {tuple_str(b)}. "
                        "If X + Y uses broadcasting, what is the output shape?"
                    )
                    answer = tuple_str(out)
                    cot = (
                        "Broadcasting aligns dimensions from right to left; 1 can expand to match the other size. "
                        "Choose max compatible size per axis to get output shape."
                    )

                else:  # reshape
                    dims = [rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 6)]
                    total = dims[0] * dims[1] * dims[2]
                    d1 = rng.choice([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18])
                    while total % d1 != 0:
                        d1 = rng.choice([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18])
                    d2 = total // d1
                    old_shape = tuple(dims)
                    new_shape = (d1, d2)
                    prompt = (
                        f"A tensor with shape {tuple_str(old_shape)} is reshaped to {tuple_str(new_shape)}. "
                        "Is the reshape valid, and what is the resulting shape?"
                    )
                    answer = f"valid; shape {tuple_str(new_shape)}"
                    cot = (
                        "A reshape is valid if total element count stays constant. "
                        "Compute both products and confirm they match."
                    )

                difficulty = "hard"

            rows.append(
                Row(
                    id=row_id,
                    split="train",
                    category="advanced_math",
                    subcategory=subcategory,
                    difficulty=difficulty,
                    prompt=prompt,
                    thinking_cot=cot,
                    answer=json.dumps(answer) if isinstance(answer, list) else fmt_num(answer) if isinstance(answer, float) else str(answer),
                )
            )
            row_id += 1

    return rows


def build_general_chat(start_id: int, count: int, rng: random.Random) -> List[Row]:
    rows: List[Row] = []

    greeting_prompts = [
        "Hi",
        "Hello there!",
        "Good morning!",
        "Hey, how are you?",
        "Yo, what is up?",
        "Can you greet me politely?",
        "Hi assistant, nice to meet you.",
        "Hello, I just joined this chat.",
    ]

    greeting_replies = [
        "Hello! Great to meet you. How can I help today?",
        "Hi there. I am ready to help with questions, planning, or coding.",
        "Good to see you. Tell me what you want to work on.",
        "Hey! I am here and ready whenever you are.",
    ]

    explanation_topics = {
        "why is the sky blue": "Sunlight scatters in the atmosphere. Shorter blue wavelengths scatter more strongly than red wavelengths, so the sky looks blue from most viewing angles.",
        "why do we have day and night": "Earth rotates on its axis. The side facing the Sun has daytime and the opposite side has nighttime.",
        "how rainbows form": "A rainbow appears when sunlight enters raindrops, refracts, reflects internally, and refracts again. Different wavelengths leave at slightly different angles, separating colors.",
        "why leaves change color": "In autumn, chlorophyll breaks down as daylight decreases. Other pigments like carotenoids and anthocyanins become more visible.",
        "how airplanes fly": "Wings move air to create pressure differences and deflect airflow downward. Lift from this flow plus engine thrust allows sustained flight.",
        "what causes tides": "Tides are mainly caused by the Moon's gravity and, to a lesser extent, the Sun. Earth's rotation and coastal geometry shape local tide patterns.",
        "what is machine learning": "Machine learning trains models on data so they can make predictions or decisions without being explicitly programmed for every rule.",
        "what is an API": "An API is an interface that lets one software system call functions or exchange data with another system using defined rules.",
    }

    practical_requests = [
        "Give me a 3-step plan to learn Python.",
        "How can I improve focus while studying?",
        "How do I prepare for a technical interview in 2 weeks?",
        "What is a good routine for writing every day?",
        "How can I organize a small team project effectively?",
        "Give me a checklist before launching a web app.",
        "How do I communicate better in meetings?",
        "What should I do if I keep procrastinating?",
    ]

    practical_responses = [
        "Define a narrow goal, time-box daily sessions, and track progress weekly. Keep a short review loop so you adjust quickly instead of guessing.",
        "Break work into 25-50 minute blocks, remove obvious distractions, and start with one high-impact task. Consistency beats intensity over time.",
        "Prioritize fundamentals first, then practice realistic problems under time limits. Finish with mock sessions and post-mortem notes.",
        "Plan tomorrow the night before, commit to a small first step, and use accountability. Momentum is easier to maintain than to recover.",
    ]

    friendly_questions = [
        "Can you tell me a fun fact?",
        "What is a healthy way to start the day?",
        "Give me a short motivational line.",
        "How do I calm down before an exam?",
        "Any tip for better sleep habits?",
        "What is a good way to learn faster?",
        "How can I become better at communication?",
        "Suggest a weekend mini-project idea.",
    ]

    friendly_responses = [
        "Try a tiny habit: start with two focused minutes and expand only after it feels automatic.",
        "Use a simple routine: breathe slowly, define one task, then begin before overthinking.",
        "Make learning active: explain ideas in your own words and test recall without notes.",
        "Good communication improves when you ask clarifying questions and summarize decisions clearly.",
        "A practical mini-project is building a small tracker app with local storage and clean UI.",
    ]

    # Keep exact count distribution: 100 greetings, 100 explanations, 100 practical, 100 friendly.
    row_id = start_id
    for i in range(100):
        prompt = greeting_prompts[i % len(greeting_prompts)]
        answer = greeting_replies[rng.randrange(len(greeting_replies))]
        cot = "Identify greeting intent, mirror friendly tone, and invite the next user need in one concise reply."
        rows.append(
            Row(
                id=row_id,
                split="train",
                category="general_chat",
                subcategory="greeting",
                difficulty="easy",
                prompt=prompt,
                thinking_cot=cot,
                answer=answer,
            )
        )
        row_id += 1

    explanation_keys = list(explanation_topics.keys())
    for i in range(100):
        topic = explanation_keys[i % len(explanation_keys)]
        prompt = f"Explain {topic}."
        answer = explanation_topics[topic]
        cot = (
            "Classify this as an explanatory question, provide a causal mechanism in plain language, "
            "and avoid unnecessary jargon."
        )
        rows.append(
            Row(
                id=row_id,
                split="train",
                category="general_chat",
                subcategory="explanation",
                difficulty="medium",
                prompt=prompt,
                thinking_cot=cot,
                answer=answer,
            )
        )
        row_id += 1

    for i in range(100):
        prompt = practical_requests[i % len(practical_requests)]
        answer = practical_responses[rng.randrange(len(practical_responses))]
        cot = (
            "Treat as practical coaching. Provide actionable, ordered guidance that can be applied immediately."
        )
        rows.append(
            Row(
                id=row_id,
                split="train",
                category="general_chat",
                subcategory="practical_advice",
                difficulty="medium",
                prompt=prompt,
                thinking_cot=cot,
                answer=answer,
            )
        )
        row_id += 1

    for i in range(100):
        prompt = friendly_questions[i % len(friendly_questions)]
        answer = friendly_responses[rng.randrange(len(friendly_responses))]
        cot = "Detect supportive tone request, keep response concise, positive, and directly useful to the user context."
        rows.append(
            Row(
                id=row_id,
                split="train",
                category="general_chat",
                subcategory="friendly_chat",
                difficulty="easy",
                prompt=prompt,
                thinking_cot=cot,
                answer=answer,
            )
        )
        row_id += 1

    if len(rows) != count:
        raise ValueError("General chat generation did not create expected row count.")

    return rows


def reverse_string_code(language: str) -> str:
    if language == "python":
        return """```python
def reverse_string(s: str) -> str:
    return s[::-1]
```
Use slicing with a step of `-1` to reverse the string."""
    if language == "javascript":
        return """```javascript
function reverseString(str) {
  return str.split('').reverse().join('');
}
```
Split into characters, reverse the array, then join back."""
    return """```typescript
function reverseString(str: string): string {
  return [...str].reverse().join('');
}
```
Spread handles unicode code points better than simple split."""


def landing_page_snippet(theme: str) -> str:
    return f"""```html
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{theme.title()} Landing</title>
  <style>
    body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: linear-gradient(120deg, #f8fafc, #e2e8f0); }}
    .hero {{ max-width: 900px; margin: 0 auto; padding: 72px 24px; text-align: center; }}
    h1 {{ font-size: clamp(2rem, 5vw, 3.5rem); margin-bottom: 16px; }}
    p {{ color: #334155; line-height: 1.6; }}
    button {{ margin-top: 20px; padding: 12px 20px; border: none; border-radius: 10px; background: #0f172a; color: white; cursor: pointer; }}
  </style>
</head>
<body>
  <section class=\"hero\">
    <h1>{theme.title()} for Modern Teams</h1>
    <p>Ship faster with a clear workflow, measurable outcomes, and collaboration built in.</p>
    <button id=\"cta\">Get Started</button>
  </section>
  <script>
    document.getElementById('cta').addEventListener('click', () => alert('Welcome!'));
  </script>
</body>
</html>
```
Single-file landing page with semantic layout, responsive typography, and a CTA interaction."""


def portfolio_snippet(stack: str) -> str:
    return f"""```html
<section class=\"portfolio\">
  <h2>Projects</h2>
  <article>
    <h3>{stack} Dashboard</h3>
    <p>Built analytics widgets, auth flow, and role-based routing.</p>
  </article>
  <article>
    <h3>Realtime Chat</h3>
    <p>Implemented presence, typing indicators, and message persistence.</p>
  </article>
</section>
```
Use this section inside a portfolio page and style it with CSS grid for responsive cards."""


def react_component_snippet(component_name: str) -> str:
    return f"""```tsx
import {{ useState }} from 'react';

type Props = {{ title: string }};

export function {component_name}({{ title }}: Props) {{
  const [open, setOpen] = useState(false);

  return (
    <section>
      <button onClick={{() => setOpen(v => !v)}}>
        {{open ? 'Hide' : 'Show'}} {{title}}
      </button>
      {{open && <p>This is a reusable UI block with local state.</p>}}
    </section>
  );
}}
```
Typed props and local state keep the component predictable and reusable."""


def generic_snippet(topic: str) -> str:
    if topic == "debounce_js":
        return """```javascript
function debounce(fn, delay = 300) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}
```
Debounce limits rapid repeated calls (search input, resize events)."""
    if topic == "fetch_retry_py":
        return """```python
import time
import requests

def get_with_retry(url: str, retries: int = 3, timeout: int = 10):
    for attempt in range(1, retries + 1):
        try:
            res = requests.get(url, timeout=timeout)
            res.raise_for_status()
            return res.json()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(attempt)
```
Retry with incremental backoff for transient network failures."""
    if topic == "array_flat_ts":
        return """```typescript
function flatten<T>(input: T[][]): T[] {
  return input.reduce((acc, cur) => acc.concat(cur), [] as T[]);
}
```
Generic helper to flatten one nesting level in TypeScript."""
    return """```javascript
function groupBy(items, keyFn) {
  return items.reduce((acc, item) => {
    const key = keyFn(item);
    (acc[key] ||= []).push(item);
    return acc;
  }, {});
}
```
`groupBy` collects items into object buckets by a computed key."""


def build_coding_tasks(start_id: int, count: int, rng: random.Random) -> List[Row]:
    rows: List[Row] = []

    plan = [
        ("reverse_string", 40),
        ("landing_page", 30),
        ("portfolio_section", 30),
        ("react_component", 30),
        ("generic_snippet", 40),
        ("algorithmic_task", 30),
    ]

    if sum(x for _, x in plan) != count:
        raise ValueError("Coding category plan does not match requested count.")

    languages = ["python", "javascript", "typescript"]
    themes = ["analytics", "fitness", "travel", "education", "finance", "design"]
    stacks = ["React + Node", "Vue + Firebase", "Django + PostgreSQL", "Next.js + Prisma"]
    component_names = ["ToggleCard", "AccordionItem", "NoticePanel", "StatusChip", "InfoDrawer"]
    snippet_topics = ["debounce_js", "fetch_retry_py", "array_flat_ts", "group_by_js"]

    row_id = start_id
    for subcategory, qty in plan:
        for _ in range(qty):
            difficulty = "medium"

            if subcategory == "reverse_string":
                lang = languages[rng.randrange(len(languages))]
                prompt = f"Write a {lang} function to reverse a string efficiently."
                answer = reverse_string_code(lang)
                cot = (
                    "Select idiomatic syntax for the requested language and keep the function minimal, "
                    "readable, and correct for common input."
                )

            elif subcategory == "landing_page":
                theme = themes[rng.randrange(len(themes))]
                prompt = f"Create a simple {theme} landing page using vanilla HTML, CSS, and JS."
                answer = landing_page_snippet(theme)
                cot = (
                    "Provide a complete single-file example with semantic HTML, responsive CSS, "
                    "and one interactive JS behavior."
                )
                difficulty = "hard"

            elif subcategory == "portfolio_section":
                stack = stacks[rng.randrange(len(stacks))]
                prompt = f"Generate a clean portfolio projects section for a developer who uses {stack}."
                answer = portfolio_snippet(stack)
                cot = (
                    "Use concise, professional content and structure it so it can be dropped into an existing page."
                )

            elif subcategory == "react_component":
                name = component_names[rng.randrange(len(component_names))]
                prompt = f"Build a reusable React + TypeScript component named {name} with toggle state."
                answer = react_component_snippet(name)
                cot = (
                    "Define typed props, local state, and a tiny interaction pattern that demonstrates reuse."
                )
                difficulty = "hard"

            elif subcategory == "generic_snippet":
                topic = snippet_topics[rng.randrange(len(snippet_topics))]
                prompt = "Provide a production-friendly utility snippet with a short explanation."
                answer = generic_snippet(topic)
                cot = (
                    "Pick a commonly needed utility, keep implementation compact, and include a practical usage note."
                )

            else:  # algorithmic_task
                n = rng.randint(5, 15)
                prompt = f"Write code to return FizzBuzz values from 1 to {n} in JavaScript."
                answer = f"""```javascript
function fizzBuzz(n) {{
  const out = [];
  for (let i = 1; i <= n; i++) {{
    if (i % 15 === 0) out.push('FizzBuzz');
    else if (i % 3 === 0) out.push('Fizz');
    else if (i % 5 === 0) out.push('Buzz');
    else out.push(String(i));
  }}
  return out;
}}
```
Call `fizzBuzz({n})` to generate the sequence."""
                cot = (
                    "Iterate from 1..n and prioritize divisibility by 15 before 3 and 5 to avoid branching mistakes."
                )

            rows.append(
                Row(
                    id=row_id,
                    split="train",
                    category="coding_tasks",
                    subcategory=subcategory,
                    difficulty=difficulty,
                    prompt=prompt,
                    thinking_cot=cot,
                    answer=answer,
                )
            )
            row_id += 1

    return rows


def generate_dataset(seed: int = 42) -> List[Row]:
    rng = random.Random(seed)
    rows: List[Row] = []

    rows.extend(build_basic_arithmetic(start_id=1, count=200, rng=rng))
    rows.extend(build_advanced_math(start_id=201, count=200, rng=rng))
    rows.extend(build_general_chat(start_id=401, count=400, rng=rng))
    rows.extend(build_coding_tasks(start_id=801, count=200, rng=rng))

    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 rows but found {len(rows)}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Thinking CoT 1K dataset.")
    parser.add_argument(
        "--output",
        default="data/train.jsonl",
        help="Output JSONL path (default: data/train.jsonl)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows = generate_dataset(seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    category_counts: Dict[str, int] = {}
    for row in rows:
        category_counts[row.category] = category_counts.get(row.category, 0) + 1

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "total_rows": len(rows),
        "category_counts": category_counts,
        "schema": [
            "id",
            "split",
            "category",
            "subcategory",
            "difficulty",
            "prompt",
            "thinking_cot",
            "answer",
            "source",
        ],
    }

    with Path("data/metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(json.dumps(category_counts, indent=2))


if __name__ == "__main__":
    main()
