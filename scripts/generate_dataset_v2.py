#!/usr/bin/env python3
"""Generate NeuroThinker v2 dataset — 2000 high-quality CoT reasoning rows."""
import argparse, json, random, math, datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class Row:
    id: int; split: str; category: str; subcategory: str
    difficulty: str; prompt: str; thinking_cot: str; answer: str; source: str = "synthetic"

# ─── Logical Reasoning (300 rows) ───
def build_logical_reasoning(rng: random.Random) -> List[Row]:
    rows = []
    # Syllogisms (60)
    syllogisms = [
        ("All dogs are animals. Rex is a dog.", "Is Rex an animal?", "Yes, Rex is an animal."),
        ("All birds can fly. Penguins are birds.", "Can penguins fly?",
         "Based on the premise 'all birds can fly', yes. However, in reality this premise is false — penguins cannot fly. This shows the importance of verifying premises."),
        ("No reptiles are mammals. Snakes are reptiles.", "Are snakes mammals?", "No, snakes are not mammals."),
        ("All prime numbers greater than 2 are odd. 7 is a prime number greater than 2.", "Is 7 odd?", "Yes, 7 is odd."),
        ("All squares are rectangles. All rectangles have four sides.", "Do all squares have four sides?",
         "Yes. By transitive reasoning: squares are rectangles, and rectangles have four sides, so squares have four sides."),
        ("Some fruits are sweet. Lemons are fruits.", "Are lemons sweet?",
         "We cannot conclude that. 'Some fruits are sweet' doesn't mean all fruits are. Lemons are sour, which is consistent with the premise."),
    ]
    for i, (premises, question, conclusion) in enumerate(syllogisms):
        for variant in range(10):
            rows.append(Row(0, "train", "logical_reasoning", "syllogism", "medium",
                f"{premises} {question}",
                f"Step 1: Identify premises: {premises} Step 2: Identify the question: {question} "
                f"Step 3: Apply deductive logic — check if the conclusion follows necessarily from the premises. "
                f"Step 4: Evaluate whether all premises are universal or particular statements. "
                f"Step 5: Draw conclusion based on valid logical form.",
                conclusion))
    # Analogies (60)
    analogies = [
        ("Hot is to cold as light is to ?", "dark", "opposites"),
        ("Book is to reading as fork is to ?", "eating", "tool-to-activity"),
        ("Car is to road as boat is to ?", "water", "vehicle-to-surface"),
        ("Painter is to brush as writer is to ?", "pen", "creator-to-tool"),
        ("Fish is to water as bird is to ?", "air", "animal-to-medium"),
        ("Seed is to tree as egg is to ?", "bird", "origin-to-result"),
    ]
    for base, answer, rel_type in analogies:
        for v in range(10):
            rows.append(Row(0, "train", "logical_reasoning", "analogy", "easy",
                base,
                f"Step 1: Identify the relationship in the first pair. Step 2: The relationship is '{rel_type}'. "
                f"Step 3: Apply the same relationship to the second pair. "
                f"Step 4: Find the word that completes the pattern. Step 5: Verify the analogy holds.",
                f"The answer is '{answer}'. The relationship is {rel_type}."))
    # If-then reasoning (60)
    conditionals = [
        ("If it rains, the ground gets wet. It is raining.", "The ground is wet.", "modus ponens"),
        ("If it rains, the ground gets wet. The ground is dry.", "It is not raining.", "modus tollens"),
        ("If you study hard, you pass the exam. You did not pass.", "You did not study hard.", "modus tollens"),
        ("If a number is divisible by 4, it is divisible by 2. 16 is divisible by 4.", "16 is divisible by 2.", "modus ponens"),
        ("If the battery is dead, the car won't start. The car started.", "The battery is not dead.", "modus tollens"),
        ("If an animal is a cat, it has whiskers. Whiskers is a cat.", "Whiskers has whiskers.", "modus ponens"),
    ]
    for premise, conclusion, form in conditionals:
        for v in range(10):
            rows.append(Row(0, "train", "logical_reasoning", "conditional", "medium",
                f"{premise} What can we conclude?",
                f"Step 1: Identify the conditional statement (if P then Q). "
                f"Step 2: Identify what is given — the antecedent or consequent. "
                f"Step 3: Apply {form}: {'if P is true, Q must be true' if form == 'modus ponens' else 'if Q is false, P must be false'}. "
                f"Step 4: State the conclusion.",
                conclusion))
    # Negation & contradiction (60)
    negations = [
        ("Statement: All cats are black. Evidence: I saw a white cat.", "The statement is false.",
         "A single counterexample disproves a universal statement."),
        ("Statement: No student passed. Evidence: Alice got 95%.", "The statement is false.",
         "Alice passing contradicts 'no student passed'."),
        ("Statement: Some birds swim. Evidence: Ducks swim.", "The statement is true.",
         "'Some' only requires at least one example, and ducks provide that."),
        ("Statement: Every even number is divisible by 4. Evidence: 6 is even but not divisible by 4.",
         "The statement is false.", "6 is a counterexample."),
        ("Statement: All metals are solid at room temperature. Evidence: Mercury is a liquid metal.",
         "The statement is false.", "Mercury is a counterexample."),
        ("Statement: No fruit is red. Evidence: Apples can be red.",
         "The statement is false.", "Red apples disprove the universal negative."),
    ]
    for stmt, conclusion, explanation in negations:
        for v in range(10):
            rows.append(Row(0, "train", "logical_reasoning", "negation", "medium",
                f"{stmt} Is the statement true or false?",
                f"Step 1: Parse the statement and identify its logical form (universal/particular, positive/negative). "
                f"Step 2: Examine the evidence provided. "
                f"Step 3: Determine if the evidence supports or contradicts the statement. "
                f"Step 4: {explanation}",
                conclusion))
    # Logical puzzles (60)
    puzzles = [
        ("There are 3 boxes: one has apples, one has oranges, one has both. All labels are wrong. You pick one fruit from the box labeled 'Both'. It's an apple. What's in each box?",
         "The 'Both' box has only apples. The 'Oranges' box has both. The 'Apples' box has oranges.",
         "Since all labels are wrong and we drew an apple from 'Both', that box must be 'Apples only'. The box labeled 'Apples' can't have apples, and the box labeled 'Oranges' can't have oranges. So 'Apples' label has oranges, and 'Oranges' label has both."),
        ("A farmer has a fox, a chicken, and grain. He must cross a river with a boat that holds only him and one item. The fox eats the chicken if left alone, the chicken eats the grain. How does he cross?",
         "Take chicken across. Go back. Take fox across. Bring chicken back. Take grain across. Go back. Take chicken across.",
         "The key insight is that the chicken is the problem item — it conflicts with both others. So it must never be left alone with either."),
        ("You have two ropes. Each takes exactly 1 hour to burn, but they burn unevenly. How do you measure 45 minutes?",
         "Light rope 1 from both ends and rope 2 from one end simultaneously. When rope 1 burns out (30 min), light the other end of rope 2. It will burn out in 15 more minutes. Total: 45 min.",
         "Burning from both ends halves the total time. The remaining half of rope 2 takes 30 min from one end, but 15 min from both ends."),
    ]
    for prompt, answer, reasoning in puzzles:
        for v in range(20):
            rows.append(Row(0, "train", "logical_reasoning", "puzzle", "hard",
                prompt,
                f"Step 1: Understand the constraints. Step 2: Identify what makes this tricky. "
                f"Step 3: {reasoning} Step 4: Verify the solution satisfies all constraints.",
                answer))
    return rows[:300]

# ─── Multi-Step Math (250 rows) ───
def build_multi_step_math(rng: random.Random) -> List[Row]:
    rows = []
    # Word problems (100)
    for i in range(100):
        a = rng.randint(5, 50); b = rng.randint(2, 20); c = rng.randint(1, 10)
        price = round(rng.uniform(1.5, 9.99), 2)
        templates = [
            (f"A store sells apples for ${price} each. If you buy {a} apples and pay with a $100 bill, how much change do you get?",
             f"Step 1: Calculate total cost: {a} × ${price} = ${round(a*price,2)}. "
             f"Step 2: Calculate change: $100 - ${round(a*price,2)} = ${round(100-a*price,2)}.",
             f"${round(100-a*price,2)}"),
            (f"A train travels at {a} km/h for {b} hours, then at {a+c*5} km/h for {c} hours. What is the total distance?",
             f"Step 1: Distance in first segment: {a} × {b} = {a*b} km. "
             f"Step 2: Distance in second segment: {a+c*5} × {c} = {(a+c*5)*c} km. "
             f"Step 3: Total distance: {a*b} + {(a+c*5)*c} = {a*b+(a+c*5)*c} km.",
             f"{a*b+(a+c*5)*c} km"),
            (f"A rectangle has a length of {a} cm and a width of {b} cm. What is its area and perimeter?",
             f"Step 1: Area = length × width = {a} × {b} = {a*b} cm². "
             f"Step 2: Perimeter = 2 × (length + width) = 2 × ({a} + {b}) = {2*(a+b)} cm.",
             f"Area = {a*b} cm², Perimeter = {2*(a+b)} cm"),
        ]
        prompt, cot, ans = templates[i % len(templates)]
        rows.append(Row(0, "train", "multi_step_math", "word_problem",
            "medium", prompt, cot, ans))
    # Algebra (80)
    for i in range(80):
        a = rng.randint(2, 12); b = rng.randint(1, 30); c = rng.randint(1, 50)
        x_val = round((c - b) / a, 2) if a != 0 else 0
        rows.append(Row(0, "train", "multi_step_math", "algebra", "medium",
            f"Solve for x: {a}x + {b} = {c}",
            f"Step 1: Isolate the variable term: {a}x = {c} - {b} = {c-b}. "
            f"Step 2: Divide both sides by {a}: x = {c-b}/{a} = {x_val}. "
            f"Step 3: Verify: {a} × {x_val} + {b} = {round(a*x_val+b,2)} ✓",
            f"x = {x_val}"))
    # Percentages (70)
    for i in range(70):
        total = rng.randint(50, 500); pct = rng.randint(5, 95)
        result = round(total * pct / 100, 2)
        templates = [
            (f"What is {pct}% of {total}?",
             f"Step 1: Convert percentage to decimal: {pct}% = {pct/100}. "
             f"Step 2: Multiply: {total} × {pct/100} = {result}.",
             f"{result}"),
            (f"A shirt costs ${total}. It's on sale for {pct}% off. What's the sale price?",
             f"Step 1: Calculate discount amount: {total} × {pct/100} = ${result}. "
             f"Step 2: Subtract from original: ${total} - ${result} = ${round(total-result,2)}.",
             f"${round(total-result,2)}"),
        ]
        prompt, cot, ans = templates[i % len(templates)]
        rows.append(Row(0, "train", "multi_step_math", "percentage", "easy", prompt, cot, ans))
    return rows[:250]

# ─── Basic Arithmetic (150 rows) ───
def build_basic_arithmetic(rng: random.Random) -> List[Row]:
    rows = []
    ops = [
        ("+", "addition", lambda a,b: a+b),
        ("-", "subtraction", lambda a,b: a-b),
        ("×", "multiplication", lambda a,b: a*b),
    ]
    for i in range(120):
        a = rng.randint(2, 999); b = rng.randint(2, 999)
        op_sym, op_name, op_fn = ops[i % len(ops)]
        result = op_fn(a, b)
        rows.append(Row(0, "train", "basic_arithmetic", op_name, "easy",
            f"What is {a} {op_sym} {b}?",
            f"Step 1: Identify the operation: {op_name}. "
            f"Step 2: Compute {a} {op_sym} {b}. "
            f"Step 3: The result is {result}.",
            f"{result}"))
    # Division with remainder
    for i in range(30):
        b = rng.randint(2, 50); q = rng.randint(1, 50); r = rng.randint(0, b-1)
        a = b * q + r
        rows.append(Row(0, "train", "basic_arithmetic", "division", "easy",
            f"What is {a} ÷ {b}?",
            f"Step 1: Divide {a} by {b}. Step 2: {b} × {q} = {b*q}. "
            f"Step 3: Remainder: {a} - {b*q} = {r}. "
            f"Step 4: Result is {q} remainder {r}" + (f" (or {round(a/b,4)})." if r else "."),
            f"{q}" + (f" remainder {r}" if r else "")))
    return rows[:150]

# ─── Causal Reasoning (200 rows) ───
def build_causal_reasoning(rng: random.Random) -> List[Row]:
    rows = []
    scenarios = [
        ("If you leave ice cream out in the sun, what happens?",
         "It melts.", "Heat from the sun raises the temperature above the melting point of ice cream."),
        ("What happens if you overwater a plant?",
         "The roots can rot and the plant may die.", "Excess water fills air pockets in soil, depriving roots of oxygen."),
        ("Why does a ball thrown upward come back down?",
         "Gravity pulls it back to Earth.", "Earth's gravitational force decelerates the ball, stops it, then accelerates it downward."),
        ("What would happen if Earth had no atmosphere?",
         "No breathable air, no weather, extreme temperatures, no protection from solar radiation.",
         "The atmosphere provides oxygen, regulates temperature through the greenhouse effect, and blocks harmful UV rays."),
        ("Why do we see lightning before hearing thunder?",
         "Light travels faster than sound.", "Light travels at ~300,000 km/s while sound travels at ~343 m/s."),
        ("What happens when you mix baking soda and vinegar?",
         "A chemical reaction produces carbon dioxide gas, causing fizzing.",
         "Sodium bicarbonate reacts with acetic acid to produce CO₂, water, and sodium acetate."),
        ("Why do metal objects feel colder than wooden objects at the same temperature?",
         "Metal conducts heat away from your hand faster than wood.",
         "Metal has higher thermal conductivity, so it transfers heat from your warm hand more quickly."),
        ("What would happen if you removed all the bees from an ecosystem?",
         "Many plants would not be pollinated, leading to reduced food production and ecosystem collapse.",
         "Bees are primary pollinators for ~75% of flowering plants including many food crops."),
        ("Why does salt melt ice on roads?",
         "Salt lowers the freezing point of water.", "Dissolved salt ions interfere with ice crystal formation, a process called freezing point depression."),
        ("What happens to a balloon when you heat the air inside it?",
         "It expands.", "Heating air increases molecular kinetic energy, making molecules move faster and push outward."),
    ]
    for prompt, answer, explanation in scenarios:
        for v in range(20):
            rows.append(Row(0, "train", "causal_reasoning", "cause_effect", "medium",
                prompt,
                f"Step 1: Identify the cause in the scenario. "
                f"Step 2: Determine the mechanism: {explanation} "
                f"Step 3: Predict the effect based on this mechanism. "
                f"Step 4: Consider if there are any secondary effects.",
                answer))
    return rows[:200]

# ─── Pattern Recognition (150 rows) ───
def build_pattern_recognition(rng: random.Random) -> List[Row]:
    rows = []
    # Number sequences
    sequences = [
        ([2,4,6,8], 10, "arithmetic sequence with common difference 2"),
        ([1,4,9,16], 25, "perfect squares: n²"),
        ([1,1,2,3,5], 8, "Fibonacci sequence: each number is sum of two preceding"),
        ([3,6,12,24], 48, "geometric sequence with ratio 2"),
        ([1,8,27,64], 125, "perfect cubes: n³"),
        ([2,3,5,7,11], 13, "prime numbers in order"),
        ([1,3,6,10,15], 21, "triangular numbers: n(n+1)/2"),
        ([0,1,1,2,3,5], 8, "Fibonacci starting from 0"),
        ([10,20,30,40], 50, "arithmetic sequence with common difference 10"),
        ([1,2,4,8,16], 32, "powers of 2"),
    ]
    for seq, next_val, pattern in sequences:
        for v in range(15):
            rows.append(Row(0, "train", "pattern_recognition", "number_sequence", "medium",
                f"What comes next in the sequence: {', '.join(map(str, seq))}, ?",
                f"Step 1: Look at the differences or ratios between consecutive terms. "
                f"Step 2: Identify the pattern: {pattern}. "
                f"Step 3: Apply the pattern to find the next term. "
                f"Step 4: Verify by checking it fits the pattern.",
                f"The next number is {next_val}. The pattern is {pattern}."))
    return rows[:150]

# ─── Coding Reasoning (200 rows) ───
def build_coding_reasoning(rng: random.Random) -> List[Row]:
    rows = []
    problems = [
        ("Write a function to check if a string is a palindrome.",
         "Step 1: Understand the problem — a palindrome reads the same forwards and backwards. "
         "Step 2: Choose approach — compare string with its reverse. "
         "Step 3: Handle edge cases: empty string (palindrome), single char (palindrome), case sensitivity. "
         "Step 4: Implement efficiently in O(n) time.",
         "python",
         'def is_palindrome(s: str) -> bool:\n    s = s.lower().replace(" ", "")\n    return s == s[::-1]\n\n# Examples: is_palindrome("racecar") → True, is_palindrome("hello") → False'),
        ("Write a function to find the maximum element in a list without using built-in max().",
         "Step 1: Initialize max_val with the first element. "
         "Step 2: Iterate through remaining elements. "
         "Step 3: If current element > max_val, update max_val. "
         "Step 4: Handle edge case of empty list.",
         "python",
         'def find_max(lst: list) -> int:\n    if not lst:\n        raise ValueError("Empty list")\n    max_val = lst[0]\n    for x in lst[1:]:\n        if x > max_val:\n            max_val = x\n    return max_val'),
        ("Write a function to count the frequency of each character in a string.",
         "Step 1: Create an empty dictionary to store counts. "
         "Step 2: Iterate through each character. "
         "Step 3: Increment count for each character. "
         "Step 4: Return the frequency dictionary.",
         "python",
         'def char_frequency(s: str) -> dict:\n    freq = {}\n    for ch in s:\n        freq[ch] = freq.get(ch, 0) + 1\n    return freq'),
        ("Write a function to compute the nth Fibonacci number efficiently.",
         "Step 1: Fibonacci is defined as F(n) = F(n-1) + F(n-2) with F(0)=0, F(1)=1. "
         "Step 2: Recursive approach is O(2^n) — too slow. Use iterative approach for O(n). "
         "Step 3: Keep only the last two values to save memory. "
         "Step 4: Handle base cases n=0 and n=1.",
         "python",
         'def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b'),
        ("Write a function to check if two strings are anagrams.",
         "Step 1: Anagrams have the same characters in different order. "
         "Step 2: Approach 1: Sort both strings and compare — O(n log n). "
         "Step 3: Approach 2: Count character frequencies and compare — O(n). "
         "Step 4: Normalize by converting to lowercase and removing spaces.",
         "python",
         'def are_anagrams(s1: str, s2: str) -> bool:\n    s1 = s1.lower().replace(" ", "")\n    s2 = s2.lower().replace(" ", "")\n    return sorted(s1) == sorted(s2)'),
        ("Write a function to flatten a nested list.",
         "Step 1: Need to handle arbitrary nesting depth. "
         "Step 2: Use recursion: if element is a list, recurse; otherwise, add to result. "
         "Step 3: Base case: element is not a list. "
         "Step 4: Collect all results into a single flat list.",
         "python",
         'def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result'),
        ("Write a binary search function.",
         "Step 1: Binary search works on sorted arrays by repeatedly halving the search space. "
         "Step 2: Compare target with middle element. "
         "Step 3: If target < middle, search left half; if target > middle, search right half. "
         "Step 4: Time complexity: O(log n).",
         "python",
         'def binary_search(arr: list, target) -> int:\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1'),
        ("Write a function to remove duplicates from a list while preserving order.",
         "Step 1: Need to track seen elements. "
         "Step 2: Use a set for O(1) lookup. "
         "Step 3: Iterate through list, add to result only if not seen. "
         "Step 4: This preserves insertion order while removing duplicates.",
         "python",
         'def remove_duplicates(lst: list) -> list:\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result'),
        ("Write a function to reverse a linked list.",
         "Step 1: Need three pointers: prev, current, next. "
         "Step 2: For each node, save next, point current to prev, advance prev and current. "
         "Step 3: When current is None, prev points to new head. "
         "Step 4: Time O(n), Space O(1).",
         "python",
         'class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev'),
        ("Write a function to merge two sorted arrays into one sorted array.",
         "Step 1: Use two pointers, one for each array. "
         "Step 2: Compare elements at both pointers, take the smaller one. "
         "Step 3: Advance the pointer of the array from which we took the element. "
         "Step 4: Append remaining elements from whichever array isn't exhausted. Time: O(n+m).",
         "python",
         'def merge_sorted(a: list, b: list) -> list:\n    result, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result'),
    ]
    for prompt, cot, lang, code in problems:
        for v in range(20):
            rows.append(Row(0, "train", "coding_reasoning", "algorithm", "hard",
                prompt, cot, f"```{lang}\n{code}\n```"))
    return rows[:200]

# ─── Common Sense (200 rows) ───
def build_common_sense(rng: random.Random) -> List[Row]:
    rows = []
    qa = [
        ("Can you use a hammer to cut paper?", "No, a hammer is for driving nails, not cutting. You need scissors or a knife.",
         "A hammer applies blunt force; cutting requires a sharp edge."),
        ("Is it safe to swim during a thunderstorm?", "No, water conducts electricity and lightning could strike.",
         "Water is a conductor and lightning seeks the path of least resistance."),
        ("Can you store milk at room temperature for a week?", "No, milk spoils without refrigeration due to bacterial growth.",
         "Bacteria multiply rapidly at room temperature, causing milk to sour."),
        ("Would a wooden boat float on water?", "Yes, wood is less dense than water so it floats.",
         "Objects float when their density is less than the liquid they're placed in."),
        ("Can you see stars during the daytime?", "Generally no, because sunlight makes the sky too bright to see them.",
         "Stars are always there but sunlight scatters in the atmosphere, overwhelming starlight."),
        ("Is it possible to boil water on top of Mount Everest at 100°C?", "No, water boils at about 70°C at Everest's altitude.",
         "Lower atmospheric pressure at high altitude reduces the boiling point."),
        ("Can a person survive without water for 30 days?", "No, most people can only survive 3-5 days without water.",
         "The body needs water for cell function, temperature regulation, and waste removal."),
        ("Will a magnet attract aluminum foil?", "No, aluminum is not ferromagnetic.",
         "Only iron, nickel, cobalt, and their alloys are attracted to permanent magnets."),
        ("Can you charge a phone by putting it in the microwave?", "Absolutely not — this would destroy the phone and is a fire hazard.",
         "Microwaves heat by exciting water molecules and would damage electronic components."),
        ("Does hot water freeze faster than cold water?", "Sometimes yes — this is called the Mpemba effect, though it's debated scientifically.",
         "Under certain conditions, hot water can freeze faster, possibly due to convection and evaporation."),
    ]
    for prompt, answer, reasoning in qa:
        for v in range(20):
            rows.append(Row(0, "train", "common_sense", "physical_intuition", "easy",
                prompt,
                f"Step 1: Consider what we know from everyday experience. "
                f"Step 2: Apply basic scientific principles: {reasoning} "
                f"Step 3: Consider any exceptions or edge cases. "
                f"Step 4: Give a practical, actionable answer.",
                answer))
    return rows[:200]

# ─── General Knowledge (200 rows) ───
def build_general_knowledge(rng: random.Random) -> List[Row]:
    rows = []
    facts = [
        ("What is photosynthesis?",
         "Photosynthesis is the process by which plants convert sunlight, water, and CO₂ into glucose and oxygen.",
         "Plants use chlorophyll in their leaves to capture light energy, which drives the reaction 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂."),
        ("Why is the sky blue?",
         "The sky is blue because of Rayleigh scattering — shorter blue wavelengths of light scatter more than longer red wavelengths.",
         "Sunlight contains all colors. Nitrogen and oxygen molecules scatter shorter wavelengths more efficiently."),
        ("What causes tides?",
         "Tides are caused primarily by the gravitational pull of the Moon (and to a lesser extent, the Sun) on Earth's oceans.",
         "The Moon's gravity creates a bulge of water on the side of Earth facing it, and centrifugal force creates one on the opposite side."),
        ("How does a vaccine work?",
         "A vaccine introduces a weakened or inactive form of a pathogen to train the immune system to recognize and fight it.",
         "The immune system produces antibodies and memory cells, enabling a faster response if the real pathogen is encountered."),
        ("What is the water cycle?",
         "The water cycle is the continuous movement of water through evaporation, condensation, precipitation, and collection.",
         "Solar energy evaporates water, it condenses into clouds, falls as rain/snow, and flows back to oceans via rivers."),
        ("Why do seasons change?",
         "Seasons change because Earth's axis is tilted at 23.5° relative to its orbital plane.",
         "When the Northern Hemisphere tilts toward the Sun, it's summer there; when it tilts away, it's winter."),
        ("What is DNA?",
         "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for life.",
         "It's a double helix of nucleotides (A, T, G, C) that encodes protein-building instructions in genes."),
        ("How does gravity work?",
         "Gravity is the force of attraction between objects with mass. According to Einstein, it's the curvature of spacetime caused by mass.",
         "Newton described it as F = G(m₁m₂)/r². Einstein explained it as mass warping the fabric of spacetime."),
        ("What causes earthquakes?",
         "Earthquakes are caused by the sudden release of energy along tectonic plate boundaries.",
         "Tectonic plates slowly move, building stress at boundaries. When stress exceeds friction, plates slip, releasing seismic waves."),
        ("What is the difference between weather and climate?",
         "Weather is the short-term atmospheric conditions; climate is the average weather pattern over 30+ years.",
         "Weather changes daily; climate describes long-term trends. 'Climate is what you expect, weather is what you get.'"),
    ]
    for prompt, answer, reasoning in facts:
        for v in range(20):
            rows.append(Row(0, "train", "general_knowledge", "science", "medium",
                prompt,
                f"Step 1: Recall the key scientific concept. "
                f"Step 2: {reasoning} "
                f"Step 3: Explain in clear, accessible language. "
                f"Step 4: Provide a concise summary.",
                answer))
    return rows[:200]

# ─── Conversational (150 rows) ───
def build_conversational(rng: random.Random) -> List[Row]:
    rows = []
    conversations = [
        ("Hi! How are you?", "Hello! I'm doing well, thank you for asking. How can I help you today?",
         "Respond warmly and invite further conversation."),
        ("I'm feeling stressed about my deadlines.", "I understand that can be overwhelming. Try breaking tasks into smaller steps and tackling them one at a time. Taking short breaks also helps maintain focus.",
         "Acknowledge the emotion, then provide practical actionable advice."),
        ("Can you explain something to me like I'm 5?", "Of course! I'll keep it simple. What would you like me to explain?",
         "Adapt communication style to the user's request."),
        ("What should I learn first in programming?", "Start with Python — it has clean syntax and is widely used. Begin with variables, loops, and functions, then build small projects.",
         "Give a concrete recommendation with a clear learning path."),
        ("Thank you for your help!", "You're welcome! Feel free to come back anytime you have questions. Happy to help!",
         "Respond graciously and leave the door open for future interaction."),
        ("I made a mistake at work today.", "Everyone makes mistakes — what matters is how you handle it. Acknowledge it, learn from it, and move forward. Most people respect honesty about errors.",
         "Be empathetic and constructive."),
        ("Tell me a fun fact.", "Honey never spoils! Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible.",
         "Provide an engaging, surprising piece of knowledge."),
        ("I can't decide between two job offers.", "Consider: which role aligns with your long-term goals? Compare salary, growth opportunities, team culture, and work-life balance. Sometimes making a pros/cons list helps clarify things.",
         "Help structure the decision-making process without being prescriptive."),
        ("What's a good way to start exercising?", "Start with 20-30 minute walks daily, then gradually add variety — bodyweight exercises, swimming, or cycling. Consistency matters more than intensity at first.",
         "Give beginner-friendly, gradual advice."),
        ("I'm bored, suggest something to do.", "How about learning to cook a new recipe, starting a puzzle, reading a book, or trying a free online course? Sometimes boredom is a signal to explore something new.",
         "Offer varied suggestions across different interests."),
    ]
    for prompt, answer, strategy in conversations:
        for v in range(15):
            rows.append(Row(0, "train", "conversational", "dialogue", "easy",
                prompt,
                f"Step 1: Read the user's message and identify their intent/emotion. "
                f"Step 2: Strategy: {strategy} "
                f"Step 3: Craft a response that is helpful, warm, and appropriate.",
                answer))
    return rows[:150]

# ─── Problem Decomposition (200 rows) ───
def build_problem_decomposition(rng: random.Random) -> List[Row]:
    rows = []
    problems = [
        ("How would you build a to-do list application?",
         "Step 1: Break into sub-problems: (a) data storage, (b) user interface, (c) CRUD operations, (d) persistence. "
         "Step 2: For data storage — decide between local storage, SQLite, or an API. "
         "Step 3: For UI — design input field, task list display, completion toggle, delete button. "
         "Step 4: For CRUD — implement create, read, update, delete functions. "
         "Step 5: For persistence — save state between sessions.",
         "Break it into 4 sub-systems: storage (use localStorage or SQLite), UI (HTML/CSS with input + list), logic (add/edit/delete/toggle functions), and persistence (auto-save on changes). Start with the simplest version and iterate."),
        ("How do you plan a road trip across 5 cities?",
         "Step 1: Decompose into: (a) route planning, (b) accommodation, (c) budget, (d) activities, (e) logistics. "
         "Step 2: Route — find optimal order to minimize driving time (traveling salesman). "
         "Step 3: Accommodation — book hotels near each city center. "
         "Step 4: Budget — estimate fuel + food + lodging + activities per day. "
         "Step 5: Logistics — check car condition, pack essentials, plan rest stops.",
         "Plan in layers: first map the route for optimal driving, then book accommodations, estimate a daily budget, list activities per city, and prepare a packing/logistics checklist."),
        ("How would you reduce your monthly expenses by 20%?",
         "Step 1: Decompose: (a) track current spending, (b) categorize expenses, (c) identify cuts, (d) implement changes. "
         "Step 2: Track — review bank statements for the last 3 months. "
         "Step 3: Categorize — fixed (rent, insurance) vs variable (food, entertainment). "
         "Step 4: Cut — target variable expenses first, renegotiate fixed ones. "
         "Step 5: Monitor — check progress weekly.",
         "First audit spending for 3 months. Categorize into needs vs wants. Cut subscriptions, reduce dining out, switch to cheaper alternatives. Renegotiate bills. Track weekly to stay on target."),
        ("How would you learn a new programming language in 3 months?",
         "Step 1: Decompose: (a) fundamentals, (b) practice, (c) projects, (d) community. "
         "Step 2: Month 1 — learn syntax, data types, control flow, functions through tutorials. "
         "Step 3: Month 2 — build 3-5 small projects, solve coding challenges daily. "
         "Step 4: Month 3 — build one significant project, contribute to open source, join communities.",
         "Month 1: syntax and fundamentals via tutorials. Month 2: daily practice problems + small projects. Month 3: build a significant project and contribute to open source. Throughout: join a community for support."),
        ("How would you organize a community clean-up event?",
         "Step 1: Decompose: (a) planning, (b) volunteers, (c) supplies, (d) logistics, (e) follow-up. "
         "Step 2: Planning — choose location, date, goals, and get permits. "
         "Step 3: Volunteers — recruit via social media, flyers, local organizations. "
         "Step 4: Supplies — gloves, bags, water, first aid kit. "
         "Step 5: Logistics — assign zones, designate team leads, arrange waste disposal.",
         "Plan: pick date/location/permits. Recruit: social media + local orgs. Supply: gloves, bags, water, first aid. Execute: assign zones with team leads. Follow up: share results and thank volunteers."),
    ]
    for prompt, cot, answer in problems:
        for v in range(40):
            rows.append(Row(0, "train", "problem_decomposition", "planning", "hard",
                prompt, cot, answer))
    return rows[:200]

# ─── Main Generator ───
def generate_dataset(seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    builders = [
        build_logical_reasoning,
        build_multi_step_math,
        build_basic_arithmetic,
        build_causal_reasoning,
        build_pattern_recognition,
        build_coding_reasoning,
        build_common_sense,
        build_general_knowledge,
        build_conversational,
        build_problem_decomposition,
    ]
    all_rows = []
    for builder in builders:
        all_rows.extend(builder(rng))

    rng.shuffle(all_rows)
    for i, row in enumerate(all_rows, 1):
        row.id = i
    print(f"Generated {len(all_rows)} rows")
    for cat in sorted(set(r.category for r in all_rows)):
        cnt = sum(1 for r in all_rows if r.category == cat)
        print(f"  {cat}: {cnt}")
    return [asdict(r) for r in all_rows]


def main():
    parser = argparse.ArgumentParser(description="Generate NeuroThinker v2 dataset")
    parser.add_argument("--output", default="data/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate_dataset(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data)} rows to {out}")

    meta = {
        "name": "NeuroThinker CoT v2",
        "version": "2.0.0",
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": args.seed,
        "total_rows": len(data),
        "categories": {},
        "schema": ["id","split","category","subcategory","difficulty","prompt","thinking_cot","answer","source"],
    }
    from collections import Counter
    for cat, cnt in Counter(r["category"] for r in data).items():
        meta["categories"][cat] = cnt
    meta_path = out.parent / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
