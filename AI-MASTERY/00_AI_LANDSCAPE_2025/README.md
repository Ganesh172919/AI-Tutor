# 00 – AI Landscape 2025

## 🎯 What You'll Learn

* The evolution of AI from base models to reasoning agents
* Key differences between model types
* Open vs Closed source models
* Popular frameworks and tools

---

## 📊 The Evolution Timeline

```
2018–2020: GPT-1/2/3 Era
└─> Base models that predict next words
└─> Few-shot learning discovered
└─> Still research-focused

2021–2022: Instruction Models
└─> Models fine-tuned to follow instructions
└─> InstructGPT, ChatGPT breakthrough
└─> Now anyone can use them

2023: Reasoning Models Emerge
└─> Chain-of-thought prompting
└─> Models that "show their work"
└─> GPT-4, Claude, PaLM 2

2024–2025: Agentic Systems
└─> Models use tools
└─> Multi-step planning
└─> Autonomy increases
└─> OpenAgents, AutoGPT, LangChain agents
```

---

## 🧩 Model Types Explained

### 1. Base LLMs
**What they are:** Raw models trained on massive text datasets

**What they do:** Complete text, predict next tokens

**Example:** GPT-3 (base), LLaMA (base)

**Think of it like:** A person who read Wikipedia but was never taught to answer questions—just to continue sentences

### 2. Instruction-Tuned Models
**What they are:** Base models + training to follow instructions

**What they do:** Answer questions, write code, summarize

**Example:** ChatGPT, Claude, Gemini

**Think of it like:** That Wikipedia reader who then went to teacher school and learned how to explain things

### 3. Reasoning-Augmented Models
**What they are:** Instruction models + explicit reasoning steps

**What they do:** Think through problems step-by-step

**Example:** GPT-4 with CoT, o1-preview, Claude with thinking

**Think of it like:** A tutor who shows their scratch work before giving you the answer

### 4. Tool-Using Agents
**What they are:** Models + ability to call functions/APIs

**What they do:** Search the web, run code, access databases

**Example:** ChatGPT with plugins, OpenAgents, LangChain agents

**Think of it like:** A researcher with access to a library, calculator, and phone

---

## 🔓 Open vs Closed Models

### Closed Models (API-only)
**Examples:** GPT-4, Claude, Gemini Pro

**Pros:**
* Most capable
* Always improving
* No infrastructure needed

**Cons:**
* Cost per token
* Rate limits
* No customization
* Privacy concerns

### Open-Source Models
**Examples:** LLaMA 3, Mistral, Qwen 2.5, Phi-3

**Pros:**
* Run locally
* Full control
* No API costs
* Customizable

**Cons:**
* Need GPUs
* Self-hosted complexity
* Usually less capable than frontier closed models

### The Trend
Open models are catching up fast. What took GPT-3.5 in 2022 can now run on a laptop with Mistral or Phi.

---

## 🛠️ Popular Tools & Frameworks (2024-2025)

### LLM Providers

| Provider | Best For | Key Models |
|----------|----------|------------|
| OpenAI | General reasoning, code | GPT-4, o1, GPT-4o |
| Anthropic | Safety, long context | Claude 3.5 Sonnet |
| Google | Multimodal, fast | Gemini 1.5 Pro/Flash |
| Meta | Open source | LLaMA 3.1/3.2 |
| Mistral | Open source, efficient | Mistral Large, 7B |
| Alibaba | Multilingual, coding | Qwen 2.5 |

### Orchestration Frameworks

**LangChain**
* Most popular
* High-level abstractions
* Good for rapid prototyping
* Can be over-engineered for simple tasks

**LangGraph**
* By LangChain team
* Graph-based agent flows
* Better for complex multi-step reasoning
* State management built-in

**OpenAgents Style**
* Modular agent design
* Planner + Executor + Verifier pattern
* More explicit control flow

**LlamaIndex**
* Focused on RAG (Retrieval-Augmented Generation)
* Best for document Q&A systems
* Great indexing and retrieval

### Deployment Tools

* **Ollama**: Run open models locally (easiest)
* **vLLM**: Fast inference server
* **TGI (Text Generation Inference)**: Hugging Face's server
* **LiteLLM**: Unified API for 100+ LLM providers

---

## 💡 Key Insight: What LLMs Actually Are

> **LLMs are not databases. They are probabilistic reasoning simulators trained on compressed world knowledge.**

This means:

1. **They don't "know" facts** – they predict what text patterns are likely
2. **They can reason** – patterns of reasoning were in their training data
3. **They hallucinate** – when unsure, they still predict something
4. **They're general-purpose** – same model can code, write, translate, explain

Think of them like this:
* Database = filing cabinet (lookup facts)
* LLM = smart person who read everything but has no filing cabinet

---

## 🔮 What's Coming Next

1. **Longer context windows** – 10M+ tokens (entire codebases)
2. **Better reasoning** – o1-style thinking becomes standard
3. **Multimodal fusion** – vision + audio + text seamlessly
4. **Smaller, faster models** – GPT-4 performance on your phone
5. **Specialized agents** – coding agents, research agents, etc.

---

## 📝 Quick Quiz

1. What's the difference between a base LLM and an instruction-tuned LLM?
2. Why do we call them "reasoning simulators" instead of "knowledge databases"?
3. Name one advantage of open-source models over closed models.

---

**Next:** [01 - How LLMs Work Deeply →](../01_HOW_LLMs_WORK_DEEPLY/README.md)
