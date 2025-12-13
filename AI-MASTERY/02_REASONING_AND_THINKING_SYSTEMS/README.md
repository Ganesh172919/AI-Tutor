# 02 – Reasoning and Thinking Systems

## 🎯 What You'll Learn

* What reasoning really means for LLMs
* Pattern completion vs symbolic reasoning
* Techniques: CoT, ToT, Graph-of-Thought
* How to design better reasoning systems
* Agent architectures (Planner, Executor, Verifier)
* Self-reflection and critic patterns

---

## 🧩 What is "Reasoning" for LLMs?

### The Fundamental Question

**Do LLMs actually reason, or do they just pattern-match?**

**Answer:** Both, and the line is blurry.

### Two Types of Reasoning

**1. Soft Reasoning (What LLMs Do)**
* Pattern-based inference
* Probabilistic associations
* "This usually leads to that"
* Example: "Dark clouds → probably rain"

**2. Hard Reasoning (What Logic Systems Do)**
* Symbolic manipulation
* Guaranteed correctness (if rules are right)
* "If A and B, then C (always)"
* Example: "2 + 2 = 4 (exactly)"

### Why LLMs Are Soft Reasoners

They learn from data, not from formal rules:
```
Saw 10,000 examples: "X causes Y"
→ Learns association
→ Predicts Y given X
→ Looks like reasoning
```

But they can make logical mistakes because they're guessing based on patterns, not computing.

---

## 🔗 Chain-of-Thought (CoT)

### The Core Idea

**Give the model space to "think" step-by-step before answering.**

### Without CoT

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
   Each can has 3 balls. How many tennis balls does he have now?

A: 11
```

Model might guess or get it wrong.

### With CoT

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
   Each can has 3 balls. How many tennis balls does he have now?

A: Let me work through this step-by-step:
   1. Roger starts with 5 balls
   2. He buys 2 cans
   3. Each can has 3 balls
   4. Balls from cans: 2 × 3 = 6
   5. Total: 5 + 6 = 11
   
   Answer: 11 balls
```

Much more reliable!

### How to Trigger CoT

**Simple prompts:**
* "Let's think step by step."
* "Show your work."
* "Explain your reasoning."

**Few-shot examples:**
Show the model 1-3 examples of step-by-step reasoning before your question.

### Why CoT Works

1. **More tokens = more computation** – model has more "thinking space"
2. **Externalized reasoning** – like using scratch paper
3. **Self-correction** – can catch mistakes while writing
4. **Learned pattern** – training data has step-by-step solutions

---

## 🌳 Tree-of-Thought (ToT)

### The Problem with CoT

CoT is linear: Step 1 → Step 2 → Step 3 → Answer

What if Step 2 was wrong? You're stuck.

### The Solution: Explore Multiple Paths

```
Problem
  │
  ├── Approach A
  │     ├── Step A1 (dead end)
  │     └── Step A2 (promising)
  │           └── Solution 1 ✓
  │
  ├── Approach B
  │     └── Step B1
  │           └── Solution 2 ✓
  │
  └── Approach C (dead end)
```

### How ToT Works

1. **Generate multiple next steps** (branches)
2. **Evaluate each step** (which looks most promising?)
3. **Explore best paths** (depth-first or breadth-first)
4. **Backtrack if stuck**
5. **Return best solution**

### Example: 24 Game

**Problem:** Use 4 numbers and +, -, ×, ÷ to make 24

**Numbers:** 4, 5, 6, 10

**ToT Process:**
```
Start: [4, 5, 6, 10]

Branch 1: 10 - 4 = 6 → [6, 5, 6]
  → 6 ÷ 6 = 1 → [1, 5]
  → 1 × 5 = 5 (fail)

Branch 2: 6 - 5 = 1 → [4, 1, 10]
  → 10 - 1 = 9 → [4, 9]
  → 4 × 9 = 36 (fail)

Branch 3: 6 ÷ (5 - 4) = 6 → [6, 10]
  → 6 + 10 = 16 (fail)

Branch 4: (10 - 6) × (5 + 4) → ERROR (more than 2 nums)

Branch 5: 10 - 6 = 4 → [4, 4, 5]
  → 4 + 4 = 8 → [8, 5]
  → 8 × 5 = 40 (fail)
  
Branch 6: 5 × 6 = 30 → [4, 30, 10]
  → 30 - 4 = 26 → [26, 10]
  → 26 - 10 = 16 (fail)

Branch 7: (5 + 6) - 10 = 1 → [4, 1]
  → ERROR (need more operations)
  
Branch 8: 10 × 6 = 60 → [4, 5, 60]
  → 60 ÷ 5 = 12 → [4, 12]
  → 12 + 4 = 16 (fail)
  
Branch 9: (10 - 6) × 5 = 20 → [4, 20]
  → 20 + 4 = 24 ✓✓✓
```

Solution: (10 - 6) × 5 + 4 = 24

### When to Use ToT

* Puzzle-solving
* Planning problems
* Multiple valid approaches exist
* Dead ends are common

---

## 🕸️ Graph-of-Thought (GoT)

### Beyond Trees

Trees are hierarchical. But reasoning can loop back, merge paths, or jump around.

**Graph-of-Thought allows:**
* Non-linear exploration
* Merging insights from different paths
* Referencing earlier thoughts

### Example: Research Synthesis

```
Topic: "Climate change impacts"

Node 1: Temperature rise
Node 2: Sea level rise
Node 3: Extreme weather
Node 4: Ecosystem disruption
Node 5: Economic impacts

Connections:
  1 → 2 (temp melts ice → sea level)
  1 → 3 (temp changes weather patterns)
  2 → 5 (flooding costs money)
  3 → 5 (storms destroy property)
  4 → 5 (crop failure)
  1 → 4 (temp stresses ecosystems)

Synthesis: All paths lead to economic impacts
         Temperature is the root cause
```

### When to Use GoT

* Complex interconnected problems
* Need to synthesize from multiple angles
* Causal reasoning

---

## 🔄 Self-Reflection and Self-Correction

### The Idea

After generating an answer, ask the model to critique itself.

### Basic Pattern

```
1. Generate initial answer
2. Ask: "Is this answer correct? What might be wrong?"
3. If model finds issues, ask it to fix them
4. Repeat until confident
```

### Example

**Question:** What's 15% of 80?

**First Answer:**
```
15% = 0.15
0.15 × 80 = 12
Answer: 12
```

**Self-Reflection Prompt:**
```
"Check your work. Is 12 correct?"
```

**Model's Response:**
```
Let me verify:
15% of 80 = 0.15 × 80 = 12
Check: 12 ÷ 80 = 0.15 = 15% ✓
Yes, 12 is correct.
```

**If it was wrong:**
```
Wait, let me recalculate:
15% of 80 = 15/100 × 80 = 1200/100 = 12
Hmm, that's the same. But let me try another way:
10% of 80 = 8
5% of 80 = 4  
15% = 10% + 5% = 8 + 4 = 12 ✓
Confirmed: 12
```

---

## 🤖 Agent Design Patterns

### The Three-Agent Pattern

Most sophisticated AI systems use this structure:

```
┌─────────────┐
│   PLANNER   │ ← "What should we do?"
└──────┬──────┘
       │ (plan)
       ▼
┌─────────────┐
│  EXECUTOR   │ ← "Do the thing"
└──────┬──────┘
       │ (result)
       ▼
┌─────────────┐
│  VERIFIER   │ ← "Is it correct?"
└──────┬──────┘
       │
       └─→ If wrong, back to Planner
```

### 1. Planner Agent

**Job:** Break problem into steps

**Input:** User goal

**Output:** List of tasks

**Example:**
```
Goal: "Book a flight to Paris"

Plan:
1. Get user's departure city
2. Get preferred dates
3. Search for flights
4. Compare prices
5. Present top 3 options
6. Confirm booking details
7. Process payment
```

### 2. Executor Agent

**Job:** Carry out tasks

**Input:** One task from plan

**Output:** Result or error

**Tools:** Can call functions, APIs, run code

**Example:**
```
Task: "Search for flights"

Actions:
- Call flight_search_api(origin="NYC", dest="CDG", date="2025-01-15")
- Receive results
- Format results

Output: [Flight options...]
```

### 3. Verifier Agent

**Job:** Check if result is correct

**Input:** Task + Result

**Output:** Pass/Fail + Feedback

**Example:**
```
Task: "Search for flights"
Result: [5 flight options]

Checks:
✓ Are all flights NYC → Paris?
✓ Are dates correct?
✓ Are prices reasonable?
✗ Missing baggage information

Verdict: FAIL - Need to get baggage info
```

---

## 🎭 Critic-Generator Pattern

### The Idea

Two models (or one model in two roles):

1. **Generator:** Creates solutions
2. **Critic:** Points out flaws

They alternate until solution is good enough.

### Example: Writing Code

**Round 1:**

Generator:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

Critic:
```
Issues:
- No input validation (what if n < 0?)
- Exponential time complexity O(2^n)
- Will crash on large n due to recursion depth
```

**Round 2:**

Generator:
```python
def fibonacci(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be non-negative integer")
    
    if n <= 1:
        return n
    
    # Use iteration instead of recursion
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```

Critic:
```
Much better!
✓ Input validation
✓ O(n) time complexity
✓ No recursion issues
Minor: Could add docstring, but functionally correct.
```

---

## 🧪 Code Example: Simple Reasoning Agent

```python
# See reasoning_agent.py for full implementation
```

---

## 💡 Key Insight

> **Reasoning improves when models are allowed to externalize thinking and verify themselves.**

This is why:
* CoT works better than direct answers
* Self-reflection catches errors
* Multi-agent systems outperform single prompts

The model isn't smarter—it just has more space to work and can catch its own mistakes.

---

## 📊 Comparison Table

| Technique | Best For | Complexity | Accuracy Gain |
|-----------|----------|------------|---------------|
| Direct Prompt | Simple Q&A | Low | Baseline |
| CoT | Math, logic | Low | +20-40% |
| ToT | Puzzles, search | Medium | +30-50% |
| GoT | Complex synthesis | High | +40-60% |
| Self-Reflection | Error-prone tasks | Medium | +15-30% |
| Multi-Agent | Open-ended goals | High | +50-70% |

---

## 🤔 Quick Quiz

1. What's the difference between soft and hard reasoning?
2. Why does Chain-of-Thought improve accuracy?
3. When would you use Tree-of-Thought instead of Chain-of-Thought?
4. What are the three roles in the classic agent pattern?

---

**Next:** [03 - Limits of AI and Open Problems →](../03_LIMITS_OF_AI_AND_OPEN_PROBLEMS/README.md)
