# 05 – Practical LLM Engineering

## 🎯 What You'll Learn

* Prompt engineering that actually works
* System prompts vs user prompts
* Context window management
* Retrieval-Augmented Generation (RAG)
* When to use which approach
* Design thinking for LLM systems

---

## ✍️ Prompt Engineering (Real Rules, Not Hacks)

### The Foundation

**Good prompts are:**
1. **Clear** - No ambiguity
2. **Specific** - Exact format/constraints
3. **Contextual** - All needed info included
4. **Testable** - You can verify if it worked

### Bad Prompt

```
"Write about dogs"
```

**Why it's bad:**
* What aspect of dogs?
* What tone?
* How long?
* What's the purpose?

### Good Prompt

```
"Write a 150-word informative paragraph about dog nutrition 
for first-time dog owners. Focus on balanced diet basics. 
Use simple language. Include 2-3 specific food examples."
```

**Why it's good:**
* Clear goal (inform first-time owners)
* Specific length (150 words)
* Defined scope (nutrition basics)
* Tone specified (simple language)
* Concrete elements (food examples)

---

## 🎯 Core Prompt Techniques

### 1. Role Assignment

**Tell the model who it is**

```
"You are an expert Python teacher with 10 years of experience 
teaching beginners. Explain concepts using simple analogies."
```

**Why it works:** Sets context for vocabulary, tone, depth

### 2. Few-Shot Examples

**Show the model what you want**

```
Example 1:
Input: "What's 25% of 80?"
Output: "25% = 1/4, so 80÷4 = 20"

Example 2:
Input: "What's 30% of 60?"
Output: "30% = 3/10, so 60×3÷10 = 18"

Now you try:
Input: "What's 15% of 200?"
```

**Why it works:** Demonstrates format, style, reasoning depth

### 3. Chain-of-Thought

**Ask for step-by-step reasoning**

```
"Solve this step-by-step. Show your work:
A train travels 180 miles in 2.5 hours. What's its speed in mph?"
```

**Output:**
```
Step 1: Identify formula → Speed = Distance ÷ Time
Step 2: Plug in values → Speed = 180 miles ÷ 2.5 hours
Step 3: Calculate → 180 ÷ 2.5 = 72
Step 4: Add units → 72 mph
```

### 4. Constraints and Format

**Specify exactly what you want**

```
"List 5 benefits of exercise.

Format:
- Each benefit in one sentence
- Start each with a verb
- No more than 15 words per benefit
- Focus on mental health benefits"
```

### 5. Negative Constraints

**Tell it what NOT to do**

```
"Explain quantum entanglement to a 10-year-old.

Do NOT:
- Use equations
- Reference wave functions
- Use jargon like 'superposition' without explanation"
```

---

## 🔧 System Prompts vs User Prompts

### System Prompt

**What it is:** Instructions that apply to entire conversation

**When to use:** Set behavior, role, rules, constraints

**Example:**
```
System: "You are a helpful coding assistant. Always:
1. Write clean, commented code
2. Explain your reasoning
3. Suggest best practices
4. Warn about common pitfalls
Never share sensitive data or unsafe code."
```

### User Prompt

**What it is:** The actual request/question

**Example:**
```
User: "Write a Python function to validate email addresses"
```

### Best Practice

```python
messages = [
    {"role": "system", "content": "You are a Python expert..."},
    {"role": "user", "content": "Write an email validator"},
    {"role": "assistant", "content": "Here's a validator..."},
    {"role": "user", "content": "Add regex for checking format"}
]
```

---

## 📏 Context Window Management

### The Problem

**Models have limited context:**
* GPT-3.5: 4K-16K tokens
* GPT-4: 8K-128K tokens
* Claude: 200K tokens
* Gemini: 1M-2M tokens

**But:** Longer context = slower + more expensive

### Strategies

#### 1. Summarization

**When context grows too large, summarize older parts**

```python
def manage_conversation(messages, max_tokens=4000):
    if count_tokens(messages) > max_tokens:
        # Keep system prompt + recent messages
        system = messages[0]
        recent = messages[-10:]
        
        # Summarize middle
        middle = messages[1:-10]
        summary = summarize(middle)
        
        messages = [system, {"role": "system", "content": summary}] + recent
    
    return messages
```

#### 2. Retrieval (RAG)

**Don't put everything in context—retrieve relevant parts**

```
User asks: "What's our refund policy?"

Instead of:
  [Entire 100-page manual in context]

Do this:
  1. Search manual for "refund"
  2. Get top 3 relevant sections
  3. Add only those to context
```

#### 3. Hierarchical Processing

**Process in stages**

```
Stage 1: Summarize each chapter individually
Stage 2: Combine summaries
Stage 3: Answer question based on combined summary
```

---

## 🔍 Retrieval-Augmented Generation (RAG)

### The Core Idea

**Don't rely on model's memory—give it sources to read**

```
┌──────────────┐
│ User Question│
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Retrieve Relevant│
│    Documents     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Add to Prompt    │
│ + Question       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   LLM Generate   │
│     Answer       │
└──────────────────┘
```

### When to Use RAG

✅ **Factual Q&A** - Answering from documents
✅ **Dynamic data** - Info that changes often
✅ **Large knowledge base** - Can't fit in context
✅ **Citations needed** - Must reference sources

❌ **Creative writing** - No source needed
❌ **Code generation** - Model already knows syntax
❌ **Simple reasoning** - No external facts required

### RAG Architecture

```python
# Simplified RAG system

def rag_query(question, knowledge_base):
    # 1. Embed question
    question_embedding = embed(question)
    
    # 2. Find similar documents
    relevant_docs = knowledge_base.similarity_search(
        question_embedding, 
        top_k=3
    )
    
    # 3. Build prompt
    context = "\n\n".join([doc.content for doc in relevant_docs])
    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
    
    # 4. Generate
    answer = llm.generate(prompt)
    
    return answer, relevant_docs  # Return sources for citation
```

### Embedding for Similarity

**How to find relevant documents?**

Convert text to numbers (embeddings), compare similarity:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed documents
docs = ["Paris is the capital of France", "Berlin is in Germany"]
doc_embeddings = model.encode(docs)

# Embed question
question = "What is the capital of France?"
question_embedding = model.encode(question)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([question_embedding], doc_embeddings)
# similarities = [[0.82, 0.31]] → First doc is more relevant
```

### Vector Databases

**Store embeddings for fast retrieval:**

| Database | Best For |
|----------|----------|
| Pinecone | Cloud, managed |
| Weaviate | Open-source, full-featured |
| Chroma | Simple, local |
| FAISS | Fast, no server |

---

## 🧠 Design Thinking for LLM Systems

### The Process

```
Problem → Decomposition → LLM Role → Architecture
```

### Example: Build a Research Assistant

**1. Problem**
```
"Help me research and summarize scientific papers"
```

**2. Decomposition**
```
Sub-tasks:
a. Find relevant papers
b. Extract key points from each
c. Synthesize into summary
d. Answer follow-up questions
```

**3. LLM Role**
```
a. Search → Use tool (not LLM)
b. Extraction → LLM with paper text
c. Synthesis → LLM with extracted points
d. Q&A → LLM with summary + RAG
```

**4. Architecture**
```
User query
  ↓
Search API (Semantic Scholar, arXiv)
  ↓
For each paper:
  LLM: Extract key findings, methods, conclusions
  ↓
LLM: Synthesize extractions into summary
  ↓
Store summary + papers in vector DB
  ↓
Q&A loop: RAG over summaries + papers
```

---

## 🎨 Prompt Templates

### Reusable Patterns

```python
templates = {
    "explain": """
Explain {topic} to a {level} learner.

Use:
- Simple language
- 1-2 analogies
- Concrete examples

Length: {length} words
""",
    
    "critique": """
Review this {type}:

{content}

Provide:
1. Strengths (2-3 points)
2. Weaknesses (2-3 points)
3. Specific improvements (3-5 actionable items)
""",
    
    "extract": """
Extract {what} from the text below.

Text:
{text}

Format: {format}
"""
}

# Usage
prompt = templates["explain"].format(
    topic="recursion",
    level="beginner programmer",
    length="200"
)
```

---

## ⚡ Performance Optimization

### 1. Streaming

**Don't wait for full response—stream tokens**

```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain AI"}],
    stream=True  # Enable streaming
)

for chunk in response:
    if 'content' in chunk['choices'][0]['delta']:
        print(chunk['choices'][0]['delta']['content'], end='')
```

**Benefits:**
* User sees progress immediately
* Feels faster
* Can stop early if needed

### 2. Caching

**Cache common queries**

```python
import hashlib
import json

cache = {}

def cached_llm_call(prompt):
    # Hash prompt for cache key
    key = hashlib.md5(prompt.encode()).hexdigest()
    
    if key in cache:
        return cache[key]
    
    # Call LLM
    response = llm.generate(prompt)
    
    # Cache it
    cache[key] = response
    return response
```

### 3. Batch Processing

**Process multiple items at once**

```python
# Instead of:
for item in items:
    result = llm.generate(f"Summarize: {item}")

# Do:
prompt = "Summarize each item below:\n\n"
for i, item in enumerate(items, 1):
    prompt += f"{i}. {item}\n"

results = llm.generate(prompt)  # One call instead of N
```

---

## 🤔 Quick Quiz

1. What makes a good prompt?
2. When should you use RAG instead of relying on model memory?
3. What's the difference between system and user prompts?
4. How do you handle context that's too large?
5. What is embedding and why is it useful for retrieval?

---

**Next:** [06 - Code Labs and Implementations →](../06_CODE_LABS_AND_IMPLEMENTATIONS/README.md)
