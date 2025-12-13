# 03 – What AI Cannot Solve (Yet)

## 🎯 What You'll Learn

* Fundamental limits of current LLMs
* Why hallucination happens
* Long-horizon planning problems
* True understanding vs simulation
* Failure modes and why they exist
* What "alignment" really means

---

## 🚫 Hard Limits of LLMs

### 1. Ground Truth Discovery

**What it means:** Finding objective facts about the world

**Why LLMs struggle:**
* They only know what was in training data
* Training data has errors and biases
* World changes after training cutoff
* Can't verify claims empirically

**Example:**
```
Q: "What's the current price of Bitcoin?"
A: [Model guesses based on training data from months ago]
   → Wrong, because Bitcoin price changes constantly
```

**What would work:** Give the model a tool to check live prices

### 2. True Understanding vs Simulation

**The philosophical question:** Do LLMs "understand" or just mimic?

**Evidence they don't truly understand:**
* Can explain quantum physics but fail simple logic puzzles
* Know grammar rules but can't apply them consistently
* Describe emotions they can't feel

**Evidence of emergent understanding:**
* Can transfer knowledge to novel situations
* Show reasoning about things never seen exactly in training
* Build coherent mental models (sometimes)

**The truth:** It's a spectrum. They have shallow, statistical understanding—not deep, conceptual understanding like humans.

### 3. Long-Horizon Planning

**The problem:** Multi-step plans that take weeks/months to execute

**Why it's hard:**
* No persistent memory across sessions (by default)
* Can't track real-world state changes
* Forgets earlier context in long conversations
* No intrinsic motivation or goals

**Example failure:**
```
Plan a 3-month project to launch a startup:

Week 1: Research market
Week 2: Build prototype
Week 3: Get funding
...
Week 12: Launch

Problem: Model can't:
- Remember Week 1 results during Week 12
- Adapt plan when Week 3 funding falls through
- Track which tasks are actually done
```

**What would work:** External memory + task tracking system + human in the loop

### 4. Value Alignment

**What it means:** Making AI do what we *actually* want (not just what we say)

**The challenge:**
```
User says: "Help me make money"
User means: "Legal, ethical ways to increase income"
Model might: Suggest scams, hacks, or shortcuts

Why? Model optimizes for literal instruction, not intent.
```

**Alignment techniques:**
* RLHF (teach preferences)
* Constitutional AI (follow principles)
* Human oversight

**But:** Still not perfect. Values are complex and subjective.

---

## 💥 Common Failure Modes

### 1. Hallucination

**What it is:** Confidently making up false information

**Why it happens:**
* Model trained to always produce output
* Gaps in knowledge → fill with plausible-sounding text
* No "I don't know" reflex (by default)

**Examples:**
```
Q: "What did Abraham Lincoln say about the internet?"
A: "In 1863, Lincoln famously remarked that 'The internet 
    will be the great equalizer of our time.'"

Reality: Internet didn't exist until 1960s.
Why model failed: Pattern-matched "Lincoln quotes" 
                   without checking dates.
```

**How to reduce:**
* Use retrieval (RAG) for facts
* Ask model to cite sources
* Lower temperature (less creativity)
* Prompt: "Say 'I don't know' if uncertain"

### 2. Overconfidence

**What it is:** Expressing certainty when wrong

**Example:**
```
Q: "Is 7919 a prime number?"
A: "Yes, definitely. 7919 is prime."

(It is prime, but model can't actually verify—just guesses)

Q: "Is 7921 a prime number?"
A: "Yes, 7921 is prime."

(Wrong! 7921 = 89 × 89)
```

**Why it happens:** Models assign probabilities to *text*, not truth

**How to reduce:**
* Ask for confidence levels
* Verify with code/tools
* Multiple samples + voting

### 3. Shortcut Learning

**What it is:** Learning spurious patterns instead of true rules

**Example from training:**
```
Training data pattern:
"The cow jumped over the ___" → moon
"The cat sat on the ___" → mat
"The dog ran to the ___" → park

Model learns: Animal + verb → common next word

Q: "The whale swam to the ___"
A: "park" (Wrong! Whales swim in ocean)

Why: Memorized patterns, didn't understand semantics
```

**Real-world example:**
```
Q: "Translate to French: 'The cat is on the mat.'"
A: "Le chat est sur le tapis." ✓

Q: "Translate to French: 'The mat is on the cat.'"
A: "Le chat est sur le tapis." ✗ (Same answer!)

Why: Model learned "cat + mat" → fixed phrase
     Didn't learn actual syntax rules
```

### 4. Context Window Limits

**What it is:** Models forget earlier conversation

**Why it happens:** Fixed context size (4K-128K tokens)

**Example:**
```
[Token 1-100]: User explains complex project
[Token 101-1000]: Discussion continues
...
[Token 10000-10500]: User asks "As I mentioned earlier..."

Model: [Has foggy memory or forgot entirely]
```

**Solutions:**
* Summarize periodically
* Use external memory/database
* Re-inject important context

### 5. Lack of Embodiment

**What it means:** No physical experience of the world

**Why it matters:**
```
Q: "Describe how to balance on a bike"
A: [Model can describe physics and technique]

But: It has never felt balance, never fallen, never learned
     through physical trial and error.

Result: Can explain intellectually but misses intuition
```

**Implication:** Poor at tasks requiring spatial/physical intuition

---

## 🔬 Why These Problems Exist

### 1. Training Data is Imperfect

* **Contains errors:** Wikipedia has mistakes
* **Contains biases:** Internet reflects human biases
* **Incomplete:** Not everything is online
* **Outdated:** Training cutoff = knowledge cutoff

### 2. Objective Mismatch

**What we want:** Helpful, truthful, harmless AI

**What it's trained on:** Predict next token accurately

**The gap:**
```
Predicting "likely next word" ≠ "correct answer"

Example:
Internet text often has: clickbait, misinformation, jokes
Model learns these patterns too
```

### 3. No Real-World Grounding

**Humans learn:**
* Through interaction (cause and effect)
* Through embodiment (physics, senses)
* Through social feedback (real-time correction)

**LLMs learn:**
* From static text
* No feedback loop during use (each conversation is independent)
* No physical grounding

**Result:** Shallow, textual knowledge without deep world models

### 4. Statistical Compression

**Training is lossy compression:**
```
Internet (100+ TB) → Model weights (100+ GB)

Compression ratio: 1000x+

Information lost: Fine details, rare facts, nuanced relationships
What's kept: Common patterns, frequent facts, general rules
```

**Implication:** Model knows "general truths" but not specific details

---

## 🧩 Open Problems in AI

### 1. Factual Accuracy

**Current state:** 80-95% accurate (depending on domain)

**Goal:** 99.9%+ (medical/legal standard)

**Challenge:** Hallucination is intrinsic to probabilistic generation

**Possible solutions:**
* Retrieval-augmented generation (RAG)
* Tool use (calculators, search engines)
* Verification models
* Hybrid symbolic-neural systems

### 2. Reasoning Reliability

**Current state:** Works on common problems, fails on edge cases

**Goal:** Provably correct reasoning

**Challenge:** Soft reasoning (statistical) vs hard reasoning (logical)

**Possible solutions:**
* Integrate with formal methods
* Neuro-symbolic AI
* Better chain-of-thought training

### 3. Long-Term Memory

**Current state:** Context windows up to 1M tokens, but expensive

**Goal:** Infinite, searchable memory

**Challenge:** Attention is O(n²) in sequence length

**Possible solutions:**
* Memory-augmented architectures
* External databases
* Hierarchical summarization

### 4. Alignment and Safety

**Current state:** RLHF helps, but not perfect

**Goal:** AI that robustly follows human values

**Challenge:** Values are complex, context-dependent, and contested

**Possible solutions:**
* Constitutional AI
* Debate between AIs
* Human oversight
* Interpretability research

### 5. True Understanding

**Current state:** Pattern matching that looks like understanding

**Goal:** Deep conceptual models

**Challenge:** We don't fully understand human understanding

**Possible solutions:**
* Multimodal learning (vision + language + action)
* Embodied AI
* Causal reasoning

---

## 🎓 Practical Takeaways

### When NOT to Use LLMs

❌ **Medical diagnosis** (too risky for errors)
❌ **Legal advice** (requires verified accuracy)
❌ **Financial predictions** (hallucination = lost money)
❌ **Safety-critical code** (bugs could be dangerous)
❌ **Real-time factual data** (training cutoff problem)

### When LLMs Excel

✅ **Drafting/brainstorming** (human reviews)
✅ **Code assistance** (with testing)
✅ **Learning/tutoring** (explaining concepts)
✅ **Content generation** (with editing)
✅ **Summarization** (with verification)

### How to Mitigate Limits

1. **Use RAG for facts** – Let model retrieve, not recall
2. **Add tools** – Calculator, search, code execution
3. **Multiple samples** – Vote or compare
4. **Human in loop** – Verify critical outputs
5. **Domain-specific fine-tuning** – For specialized tasks

---

## 🤔 Quick Quiz

1. Why do LLMs hallucinate?
2. What's the difference between simulation and true understanding?
3. Why can't LLMs do long-horizon planning well?
4. Name two ways to reduce hallucination.
5. In what domains should you NOT rely solely on LLMs?

---

## 📚 Further Reading

* "Stochastic Parrots" paper – Risks of large language models
* "Constitutional AI" paper – Anthropic's alignment approach
* "Sparks of AGI" paper – GPT-4's capabilities and limits

---

**Next:** [04 - Model Building From Scratch →](../04_MODEL_BUILDING_FROM_SCRATCH/README.md)
