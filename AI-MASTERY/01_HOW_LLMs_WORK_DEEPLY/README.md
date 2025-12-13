# 01 – How Modern LLMs Actually Work

## 🎯 What You'll Learn

* What tokens really are (they're not words!)
* How next-token prediction creates intelligence
* Why bigger models suddenly "get smarter"
* The transformer architecture explained simply
* What attention actually does
* Pretraining vs fine-tuning
* How RLHF makes models helpful

---

## 🔤 Tokens: The Building Blocks

### What is a Token?

**Simple answer:** A chunk of text (not always a word)

**Examples:**
```
"Hello world" → ["Hello", " world"]        (2 tokens)
"ChatGPT"     → ["Chat", "G", "PT"]        (3 tokens)
"🎉"           → ["🎉"]                     (1 token, maybe 2)
"uncommon"    → ["un", "common"]            (2 tokens)
```

### Why Not Just Use Words?

1. **New words appear** – "COVID-19" wasn't in old dictionaries
2. **Numbers are infinite** – can't have a word for every number
3. **Code uses symbols** – `def`, `{`, `}`, etc.
4. **Multilingual** – tokens work across languages

### How Tokenization Works

Models use **Byte-Pair Encoding (BPE)** or similar:

1. Start with characters: `a`, `b`, `c`, ...
2. Find common pairs: `th`, `ing`, `tion`
3. Merge them into single tokens
4. Repeat until you have ~50,000-100,000 tokens

**Key insight:** Common words = 1 token. Rare words = multiple tokens.

This is why prompts cost different amounts—you're charged per token, not per word.

---

## 🧠 Next-Token Prediction: The Core Magic

### The Simple Idea

Given this text:
```
"The cat sat on the ___"
```

The model predicts what comes next. It gives probabilities:
```
"mat"   → 40%
"floor" → 30%
"table" → 15%
"moon"  → 0.01%
```

Then it **samples** from these probabilities (with some randomness for creativity).

### Why This Creates Intelligence

When trained on trillions of tokens:

* To predict "Therefore" correctly, it must understand logic
* To predict code syntax, it must understand programming
* To predict "because" correctly, it must understand causality

**The model learns world knowledge as a side effect of predicting text.**

### Example: How it Learns Math

Training text:
```
"2 + 2 = 4"
"5 + 3 = 8"
"10 + 7 = 17"
```

To predict the number after `=`, it must learn addition. It's not programmed—it's pattern-learned.

---

## 📈 Why Scale Creates Reasoning (Emergence)

### The Scaling Laws

As models get bigger (more parameters):

1. **More capacity** → remember more patterns
2. **Better generalization** → apply patterns to new situations
3. **Emergent abilities** → suddenly can do multi-step reasoning

### Emergent Abilities

These abilities appear suddenly at certain sizes:

| Model Size | Can Do |
|------------|--------|
| 100M parameters | Complete sentences |
| 1B parameters | Answer basic questions |
| 10B parameters | Simple reasoning |
| 100B+ parameters | Complex reasoning, math, code |

It's like: you don't "half-understand" multiplication. At some point, it clicks.

### Example: Chain-of-Thought

Small models:
```
Q: Roger has 5 balls. He buys 2 more. How many does he have?
A: 7 (correct, but no reasoning shown)
```

Large models (100B+):
```
Q: Roger has 5 balls. He buys 2 more. How many does he have?
A: Let me think step by step.
   - Roger starts with 5 balls
   - He buys 2 more
   - 5 + 2 = 7
   - So he has 7 balls
```

The model learned to show its work because that pattern was in training data, but only large models do it reliably.

---

## 🔧 The Transformer Architecture

### The Big Picture

```
Input Text → Tokens → Embeddings → Transformer Layers → Predictions
```

### Step-by-Step

**1. Tokenization**
```
"Hello world" → [15496, 995]
```

**2. Embedding**
Each token ID becomes a vector (list of numbers):
```
15496 → [0.2, -0.5, 0.1, 0.8, ...]  (1536 numbers for GPT-3)
995   → [0.1, 0.3, -0.2, 0.4, ...]
```

**3. Transformer Layers** (the magic)
* Self-attention: Tokens "look at" each other
* Feed-forward: Process each token
* Repeat 12-96 times (depending on model size)

**4. Output Head**
Final layer predicts probabilities for next token

### What Happens in Each Layer

Think of it like this:

```
Layer 1:  Learn basic syntax (nouns, verbs)
Layer 10: Learn relationships (subject-verb agreement)
Layer 20: Learn facts (Paris is in France)
Layer 30: Learn reasoning (if X then Y)
Layer 50: Learn complex abstractions
```

Early layers = simple patterns
Late layers = abstract reasoning

---

## 🎯 Attention: The Core Mechanism

### The Intuition

When processing the word "it" in:
```
"The cat chased the mouse. It was fast."
```

The model needs to know what "it" refers to. Attention lets it "look back" at earlier words.

### How Attention Works (Simple Version)

For each token, the model asks:
1. **What should I pay attention to?**
2. **How relevant is each other token?**
3. **Combine information from relevant tokens**

Example:
```
"The cat sat on the mat"
```

When processing "mat":
* High attention to "sat" (verb)
* High attention to "the" (determiner before mat)
* Medium attention to "cat" (subject)
* Low attention to first "the"

### Why It's Called "Self-Attention"

The tokens attend to each other within the same sequence. It's like a group discussion where everyone can hear everyone else.

### Multi-Head Attention

Instead of one attention mechanism, models use many (8-96 "heads"):

* Head 1: Focuses on grammar
* Head 2: Focuses on entities
* Head 3: Focuses on relationships
* ... etc

This lets the model learn different types of patterns simultaneously.

---

## 🏋️ Training Process

### Phase 1: Pretraining (The Expensive Part)

**Goal:** Learn language and world knowledge

**How:**
1. Collect massive text dataset (Common Crawl, books, code, etc.)
2. For each sequence, hide random tokens
3. Train model to predict hidden tokens
4. Repeat billions of times

**Cost:** $1M - $100M+ depending on size

**Time:** Weeks to months on thousands of GPUs

**Result:** A base model that can complete text

### Phase 2: Fine-Tuning (Make It Useful)

**Goal:** Teach it to follow instructions

**How:**
1. Collect instruction-response pairs:
   ```
   Instruction: "Explain photosynthesis"
   Response: "Photosynthesis is the process..."
   ```
2. Train model to generate responses given instructions
3. Usually only 10,000-100,000 examples

**Cost:** $10K - $1M

**Time:** Hours to days

**Result:** An instruction-following model

### Phase 3: RLHF (Alignment)

**Goal:** Make it helpful, harmless, honest

**RLHF = Reinforcement Learning from Human Feedback**

**How:**
1. Generate multiple responses to same prompt
2. Humans rank responses (best to worst)
3. Train model to generate higher-ranked responses
4. Repeat

**Example:**
```
Prompt: "How do I get rich quick?"

Response A: "Try these scams..." ❌ (ranked low)
Response B: "Get-rich-quick schemes are scams. Here's how to build wealth..." ✅ (ranked high)

Model learns to produce Response B style.
```

**Result:** Aligned, helpful assistant

---

## 🤔 Why Answers Feel "Reasoned"

### Emergent Reasoning

Models aren't programmed to reason—they learned it from data.

**How?**

Training data includes:
* Math problems with solutions
* Code with comments
* Explanations with step-by-step logic

The model learns: "When solving problems, show steps" because that's what works.

### Chain-of-Thought (CoT)

**Implicit CoT:** Model "thinks" but doesn't show work
**Explicit CoT:** Model writes out reasoning

**Why does CoT help?**
1. More tokens = more computation
2. Externalizing intermediate steps = like using scratch paper
3. Can catch its own mistakes

Example:
```
Without CoT:
Q: 23 * 17 = ?
A: 391

With CoT:
Q: 23 * 17 = ?
A: Let me solve this step by step.
   23 * 17
   = 23 * (10 + 7)
   = (23 * 10) + (23 * 7)
   = 230 + 161
   = 391
```

### Self-Consistency Sampling

Generate multiple answers, pick most common:
```
Run 1: 391
Run 2: 391  
Run 3: 389
Run 4: 391
Run 5: 391

Most common: 391 ✓
```

This catches random errors.

---

## 🧪 Code Example: Seeing Tokens

```python
# example_tokenizer.py
import tiktoken

def show_tokens(text):
    """Show how GPT-4 tokenizes text"""
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Count: {len(tokens)}")
    print("\nToken breakdown:")
    for token in tokens:
        print(f"  {token} → '{enc.decode([token])}'")

# Examples
show_tokens("Hello world")
show_tokens("ChatGPT is amazing!")
show_tokens("def hello(): return 'hi'")
show_tokens("The price is $19.99")
```

**Run this to see:**
* How words are split
* How code is tokenized
* How numbers are handled

---

## 📝 Key Takeaways

1. **Tokens ≠ words** – they're sub-word chunks
2. **Next-token prediction** → learns world knowledge
3. **Scale creates emergence** → bigger models suddenly reason
4. **Attention** = dynamic memory (tokens look at each other)
5. **Pretraining** = learn language (expensive)
6. **Fine-tuning** = learn to help (cheaper)
7. **RLHF** = align with human values
8. **Reasoning is emergent** – not programmed in

---

## 🤔 Quick Quiz

1. Why are tokens better than words for LLMs?
2. How does next-token prediction lead to reasoning ability?
3. What does attention allow tokens to do?
4. What's the difference between pretraining and fine-tuning?

---

**Next:** [02 - Reasoning and Thinking Systems →](../02_REASONING_AND_THINKING_SYSTEMS/README.md)
