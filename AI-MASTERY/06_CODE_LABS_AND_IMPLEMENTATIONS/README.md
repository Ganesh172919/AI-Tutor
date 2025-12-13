# 06 – Code Labs and Implementations

## 🎯 What You'll Learn

This section contains **runnable code labs** that demonstrate:

1. **Minimal LLM API usage** - Get started with any provider
2. **Simple RAG system** - Build retrieval-augmented generation
3. **Reasoning agent loop** - Implement self-improving agents
4. **Self-checking agent** - Build verification systems

Each lab includes:
* Full working code
* Explanations of every part
* What to modify/experiment with
* Common pitfalls to avoid

---

## 🔧 Prerequisites

```bash
# Install dependencies
pip install openai anthropic google-generativeai
pip install sentence-transformers chromadb
pip install python-dotenv
```

```bash
# Set API keys (choose one or more)
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

---

## Lab 1: Minimal LLM API Usage

**File:** `lab1_llm_basics.py`

**What you'll learn:**
* How to call different LLM APIs
* Handle responses
* Manage conversations
* Error handling

**See the code file for complete implementation.**

---

## Lab 2: Simple RAG System

**File:** `lab2_simple_rag.py`

**What you'll learn:**
* Embed documents
* Store in vector database
* Retrieve relevant context
* Generate answers with citations

**Architecture:**
```
Documents → Chunking → Embedding → Vector DB → Retrieval → LLM + Context → Answer
```

**See the code file for complete implementation.**

---

## Lab 3: Reasoning Agent Loop

**File:** `lab3_reasoning_agent.py`

**What you'll learn:**
* Chain-of-thought prompting
* Self-reflection loops
* Multi-step problem solving
* When to stop iterating

**Agent Loop:**
```
Question → Initial Answer → Critique → Improved Answer → Verify → Done
```

**See the code file for complete implementation.**

---

## Lab 4: Self-Checking Agent

**File:** `lab4_self_checking.py`

**What you'll learn:**
* Generate multiple solutions
* Self-critique each solution
* Pick the best one
* Verify correctness

**Verification Pattern:**
```
Problem → Generate Solution → Verify → If wrong, retry → If right, return
```

**See the code file for complete implementation.**

---

## 🎓 Learning Path

**Beginner:** Start with Lab 1
**Intermediate:** Labs 2-3
**Advanced:** Lab 4

---

## 🔬 Experiments to Try

After running each lab, try:

1. **Different models** - Compare GPT-4, Claude, Gemini
2. **Different temperatures** - 0 (deterministic) to 2 (creative)
3. **Prompt variations** - See how small changes affect output
4. **Error cases** - What happens with bad input?
5. **Performance** - Time each component

---

**Next:** [07 - Revenue and Products →](../07_REVENUE_AND_PRODUCTS/README.md)
