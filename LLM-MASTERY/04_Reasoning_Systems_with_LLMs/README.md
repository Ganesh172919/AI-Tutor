# Module 04: Reasoning Systems with LLMs

## 📚 Overview

One of the most exciting developments in modern LLMs is their ability to **reason**—breaking down complex problems into steps, verifying their own work, and even using external tools. This module explores the cutting-edge techniques that give LLMs reasoning capabilities approaching human-level performance on many tasks.

From OpenAI's o1 model to chain-of-thought prompting, you'll learn how to build LLMs that don't just pattern-match, but actually *think*.

## 🎯 Learning Objectives

By the end of this module, you will:

1. **Master** Chain-of-Thought (CoT) prompting for multi-step reasoning
2. **Implement** Tree-of-Thoughts (ToT) with beam search and backtracking
3. **Build** self-consistency mechanisms for robust answers
4. **Create** tool-using agents that call APIs and execute code
5. **Design** ReAct agents that interleave reasoning and acting
6. **Develop** verification loops for self-checking and error correction
7. **Understand** how reasoning emerges in large models

## 📖 Module Contents

### 01. Chain-of-Thought Prompting (`01_chain_of_thought.md`)
- What is CoT? Breaking problems into explicit steps
- Zero-shot CoT: "Let's think step by step"
- Few-shot CoT: Providing reasoning examples
- When CoT helps vs. hurts
- Automatic CoT generation
- **Code**: CoT prompting framework with GPT-4

**Example**:
```
Problem: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

Without CoT:
Answer: 11

With CoT:
Let's solve this step-by-step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans of tennis balls
3. Each can has 3 balls, so 2 cans = 2 × 3 = 6 balls
4. Total = 5 (initial) + 6 (new) = 11 balls
Answer: 11
```

### 02. Tree-of-Thoughts (`02_tree_of_thoughts.md`)
- Beyond linear reasoning: Exploring multiple paths
- Building thought trees with beam search
- Backtracking when paths lead to dead ends
- Evaluating partial solutions
- When to use ToT vs. CoT
- **Code**: Complete ToT implementation with game playing

**ToT Example (Game of 24)**:
```
Problem: Use 4 numbers (4, 8, 8, 12) and operations (+, -, ×, ÷) to get 24

Tree of Thoughts:
Root: (4, 8, 8, 12)
├─ Branch 1: 8 ÷ 4 = 2 → (2, 8, 12)
│  ├─ 2 + 12 = 14 → (14, 8)
│  │  └─ 14 + 8 = 22 ✗ (not 24)
│  └─ 8 + 12 = 20 → (20, 2)
│     └─ 20 + 2 = 22 ✗
├─ Branch 2: 12 - 8 = 4 → (4, 4, 8)
│  └─ 4 × 4 = 16 → (16, 8)
│     └─ 16 + 8 = 24 ✓ SOLUTION!
└─ Branch 3: 8 × 8 = 64 → (64, 4, 12)
   └─ Too large, prune this branch

Solution: (12 - 8) × (8 ÷ 4) = 4 × 2 = 8... wait, wrong.
Actually: (8 ÷ 4) × 12 = 24 ✓
```

### 03. Self-Consistency (`03_self_consistency.md`)
- Sample multiple reasoning paths
- Vote on final answers
- Temperature vs. deterministic generation
- Calibrating confidence scores
- When self-consistency improves accuracy
- **Code**: Self-consistency wrapper for any LLM

**Self-Consistency Process**:
1. Generate 10 different reasoning paths (with temperature=0.7)
2. Parse final answer from each path
3. Take majority vote
4. Return most common answer

**Example**:
```
Problem: If a train travels 60 mph for 1.5 hours, how far does it go?

Path 1: 60 × 1.5 = 90 miles ✓
Path 2: 60 + 1.5 = 61.5 miles ✗
Path 3: 60 × 1.5 = 90 miles ✓
Path 4: Distance = speed × time = 60 × 1.5 = 90 ✓
Path 5: 60 / 1.5 = 40 miles ✗
Path 6: 60 × 1.5 = 90 miles ✓
...

Majority vote: 90 miles (7 out of 10 paths)
Confidence: 70%
```

### 04. Tool Use and Function Calling (`04_tool_use.md`)
- Teaching LLMs to use external tools
- Function calling APIs (OpenAI, Claude)
- Calculator, web search, code execution
- Tool selection and parameter extraction
- Error handling and retry logic
- **Code**: Multi-tool agent with Python executor

**Tool Use Example**:
```python
# Agent has access to these tools:
tools = {
    "calculator": calc_tool,
    "search": search_tool,
    "python": python_executor,
    "weather": weather_api
}

# User query
query = "What's the weather in the city where the Eiffel Tower is located?"

# Agent reasoning:
"""
I need to:
1. Use search to find where the Eiffel Tower is located
2. Use weather API to get weather for that city
"""

# Tool calls:
location = search("Where is the Eiffel Tower located?")
# → "Paris, France"

weather = weather_api(city="Paris", country="France")
# → "15°C, partly cloudy"

# Final answer:
"The weather in Paris (where the Eiffel Tower is located) is 15°C and partly cloudy."
```

### 05. ReAct: Reasoning + Acting (`05_react_paradigm.md`)
- The ReAct framework (Yao et al., 2022)
- Interleaving thought, action, observation
- Planning multi-step tasks
- Adapting plans based on observations
- **Code**: ReAct agent for web navigation

**ReAct Example**:
```
Task: Book a flight from New York to London for next Monday

Thought 1: I need to search for flights on a booking website
Action 1: Navigate to flights.com
Observation 1: On flights.com homepage

Thought 2: I need to enter departure and arrival cities
Action 2: Fill form {from: "New York", to: "London", date: "2024-12-23"}
Observation 2: Form filled, showing search button

Thought 3: Now I'll search for available flights
Action 3: Click "Search Flights"
Observation 3: Results page showing 15 flights

Thought 4: I should find the cheapest option
Action 4: Sort by price (low to high)
Observation 4: Cheapest flight is $450 with 1 stop

Thought 5: This looks good, I'll proceed to book
Action 5: Click "Select" on first result
Observation 5: Booking page opened

[Continue until task complete]
```

### 06. Verification Loops (`06_verification_loops.md`)
- Self-critique and error correction
- Multiple verification strategies
- Answer refinement through iteration
- When to stop vs. continue refining
- **Code**: Self-checking mathematical reasoner

**Verification Example**:
```
Problem: Solve the equation 2x + 5 = 13

Initial attempt:
2x + 5 = 13
2x = 8
x = 4

Verification: Let's check by substituting back
2(4) + 5 = 8 + 5 = 13 ✓ Correct!

Another problem: 3x² - 12 = 0

Initial attempt:
3x² = 12
x² = 4
x = 2

Verification: Let's check
3(2)² - 12 = 3(4) - 12 = 12 - 12 = 0 ✓

Wait, but x = -2 also works!
3(-2)² - 12 = 3(4) - 12 = 0 ✓

Refined answer: x = 2 or x = -2
```

## 💻 Code Files

| File | Description | Lines | Difficulty |
|------|-------------|-------|------------|
| `cot_prompting.py` | Chain-of-thought framework | ~300 | ⭐⭐⭐☆☆ |
| `tot_search.py` | Tree-of-thoughts with beam search | ~500 | ⭐⭐⭐⭐⭐ |
| `self_consistency.py` | Self-consistency voting | ~200 | ⭐⭐⭐☆☆ |
| `tool_agent.py` | Multi-tool agent | ~400 | ⭐⭐⭐⭐☆ |
| `react_agent.py` | ReAct framework | ~600 | ⭐⭐⭐⭐⭐ |
| `verification.py` | Self-checking reasoner | ~300 | ⭐⭐⭐⭐☆ |

## 🎯 Real-World Applications

### 1. Mathematical Reasoning
- GSM8K, MATH dataset performance
- OpenAI o1's PhD-level problem solving
- Competition programming (CodeForces)

### 2. Commonsense Reasoning
- StrategyQA, BoolQ benchmarks
- Multi-hop question answering
- Causal reasoning tasks

### 3. Code Generation and Debugging
- AlphaCode's contest performance
- GitHub Copilot's code completion
- Automated bug fixing

### 4. Scientific Discovery
- Hypothesis generation
- Experiment design
- Literature synthesis

### 5. Autonomous Agents
- WebGPT (web browsing)
- Auto-GPT (task automation)
- Research assistants (Elicit, Consensus)

## 📊 Benchmarks and Metrics

### Reasoning Benchmarks

| Benchmark | Task | Best Model (2024) | Score |
|-----------|------|-------------------|-------|
| GSM8K | Grade school math | GPT-4 + CoT | 94.2% |
| MATH | High school competition math | o1-preview | 85.5% |
| StrategyQA | Commonsense reasoning | Claude 3 Opus | 88.9% |
| HumanEval | Code generation | GPT-4 Turbo | 88.4% |
| MMLU | Multitask knowledge | GPT-4 | 86.4% |
| ARC-Challenge | Science reasoning | o1-preview | 96.7% |

### How Reasoning Improves Performance

**Without CoT (Direct Answer)**:
- GSM8K: ~50% (GPT-3)
- MATH: ~5% (GPT-3)

**With CoT Prompting**:
- GSM8K: ~80% (GPT-3.5)
- MATH: ~35% (GPT-4)

**With o1-style Extended Reasoning**:
- GSM8K: ~95%
- MATH: ~85%

**Key insight**: Reasoning time matters! More "thinking" = better performance.

## 🧠 How Reasoning Emerges

### Scale and Emergence
At small scales (<10B parameters):
- Mostly pattern matching
- Struggles with multi-step reasoning
- Fails on novel problems

At large scales (100B+ parameters):
- Emergent reasoning abilities
- Can follow multi-step logic
- Generalizes to new problem types

### Training for Reasoning

**Pre-training** (standard LM training):
- Learns patterns, not explicit reasoning
- Absorbs reasoning from web text/books
- Implicit chain-of-thought in training data

**Fine-tuning with reasoning data**:
```python
# Training examples with explicit reasoning
examples = [
    {
        "problem": "What is 15% of 200?",
        "reasoning": "To find 15% of 200:\n1. Convert 15% to decimal: 0.15\n2. Multiply: 0.15 × 200 = 30",
        "answer": "30"
    },
    # ... thousands more
]
```

**RLHF with reasoning rewards**:
- Reward models that value step-by-step reasoning
- Penalize shortcuts or incorrect logic
- Encourage showing work

### The o1 Approach

OpenAI's o1 model uses:
1. **Extended thinking time**: Doesn't rush to answer
2. **Internal chain-of-thought**: Hidden reasoning before response
3. **Self-verification**: Checks its own work
4. **Test-time scaling**: More compute → better answers

```python
# Conceptual o1 generation
def o1_generate(problem, max_thinking_tokens=10000):
    # Phase 1: Extended reasoning (hidden)
    reasoning = ""
    for _ in range(max_thinking_tokens):
        next_thought = model.generate_next_thought()
        reasoning += next_thought
        
        if is_solution_found(reasoning):
            break
    
    # Phase 2: Generate final answer
    answer = model.generate_answer(reasoning)
    
    # Phase 3: Verify
    if verify_answer(problem, answer):
        return answer
    else:
        # Try again with more thinking
        return o1_generate(problem, max_thinking_tokens * 2)
```

## 🔬 Research Frontiers (2024-2025)

### 1. Faithful Reasoning
- Ensuring LLMs actually follow their stated logic
- Detecting "reasoning shortcuts"
- Mechanistic interpretability of reasoning circuits

### 2. Multi-Modal Reasoning
- Reasoning over images, videos, code
- Diagrammatic reasoning
- Spatial reasoning

### 3. Long-Horizon Planning
- Tasks requiring 50+ steps
- Hierarchical planning
- Sub-goal decomposition

### 4. Collaborative Reasoning
- Multiple LLMs debating
- Human-AI co-reasoning
- Hybrid symbolic-neural systems

### 5. Uncertainty Quantification
- Knowing when to say "I don't know"
- Calibrated confidence scores
- Selective verification

## 💡 Practical Tips

### When to Use CoT
✅ **Use CoT when**:
- Problem requires multiple steps
- Intermediate steps are meaningful
- You want explainable answers
- Domain has clear reasoning patterns

❌ **Skip CoT when**:
- Simple lookup tasks (e.g., "What's the capital of France?")
- Ambiguous problems without clear steps
- Latency is critical (CoT adds tokens)

### Prompt Engineering for Reasoning

**Poor prompt**:
```
Solve: If John has 5 apples and gives away 2, how many does he have?
```

**Better prompt with CoT**:
```
Let's solve this step-by-step:

Problem: If John has 5 apples and gives away 2, how many does he have?

Step 1: Identify the initial quantity
Step 2: Determine the change
Step 3: Calculate the final amount
Step 4: State the answer

Solution:
```

**Best prompt with few-shot CoT**:
```
Here are some example problems solved step-by-step:

Example 1:
Problem: Mary has 8 cookies and eats 3. How many are left?
Solution:
1. Mary starts with 8 cookies
2. She eats 3 cookies
3. Remaining = 8 - 3 = 5 cookies
Answer: 5 cookies

Example 2:
Problem: A store has 20 items. 12 are sold. How many remain?
Solution:
1. Store starts with 20 items
2. 12 items are sold
3. Remaining = 20 - 12 = 8 items
Answer: 8 items

Now solve this:
Problem: If John has 5 apples and gives away 2, how many does he have?
Solution:
```

## ✏️ Exercises

### Beginner
1. Implement zero-shot CoT prompting for basic arithmetic
2. Compare CoT vs. direct answering on GSM8K subset
3. Build a simple calculator tool for LLM to use

### Intermediate
4. Implement self-consistency with voting
5. Create a ToT system for simple puzzles (e.g., Tower of Hanoi)
6. Build a multi-tool agent with search and calculator

### Advanced
7. Implement full ReAct framework for web navigation
8. Create verification loops for mathematical proofs
9. Build an agent that can break down and solve complex tasks
10. Implement o1-style extended reasoning with hidden thoughts

## 🔗 Dependencies

```bash
pip install openai anthropic  # LLM APIs
pip install langchain  # Agent frameworks
pip install sympy  # Symbolic math for verification
pip install selenium  # Web automation for ReAct
pip install beautifulsoup4  # Web scraping
```

## 📚 Required Reading

1. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** (Wei et al., 2022)
2. **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** (Yao et al., 2023)
3. **"Self-Consistency Improves Chain of Thought Reasoning in Language Models"** (Wang et al., 2022)
4. **"ReAct: Synergizing Reasoning and Acting in Language Models"** (Yao et al., 2022)
5. **"Solving Quantitative Reasoning Problems with Language Models"** (Lewkowycz et al., 2022)

## ⏱️ Estimated Time

- **Reading**: 6-8 hours
- **Coding**: 12-15 hours
- **Exercises**: 8-10 hours
- **Total**: 26-33 hours

## 🎯 Key Takeaways

After this module, you'll understand:

1. **CoT**: How explicit reasoning steps improve accuracy
2. **ToT**: When to explore multiple reasoning paths
3. **Self-Consistency**: Using sampling to boost robustness
4. **Tool Use**: Extending LLMs with external capabilities
5. **ReAct**: Interleaving thinking and acting for complex tasks
6. **Verification**: Self-checking to reduce errors

**Reasoning is the frontier of LLM capabilities. Master it, and you can build agents that solve real-world problems.**

---

**Next**: Module 05 - Code Understanding and Mindset Shift →
