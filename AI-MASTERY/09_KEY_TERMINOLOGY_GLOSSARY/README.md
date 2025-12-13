# 09 – Key Terminology Glossary

## 🎯 Purpose

Quick reference for AI/LLM terminology. Each term includes:
* Simple definition
* Why it matters
* Example usage

---

## 📚 Core Concepts

### Token
**Definition:** A chunk of text (not always a word) that LLMs process

**Why it matters:** 
* You're charged per token
* Context limits are in tokens
* "Hello world" might be 2-3 tokens depending on model

**Example:** `"ChatGPT"` → `["Chat", "G", "PT"]` (3 tokens)

---

### Embedding
**Definition:** Converting text into a list of numbers (vector) that represents meaning

**Why it matters:**
* Enables similarity search
* Powers RAG systems
* Similar meanings → similar vectors

**Example:** 
```python
"dog" → [0.2, 0.8, -0.1, ...]
"cat" → [0.3, 0.7, -0.2, ...]  # Similar to dog
"car" → [-0.5, 0.1, 0.9, ...]  # Different
```

---

### Context Window
**Definition:** Maximum amount of text an LLM can process at once

**Why it matters:**
* Limits conversation length
* Longer context = more expensive
* Must manage for long documents

**Example:** GPT-4 can handle 8K-128K tokens depending on version

---

### Temperature
**Definition:** Controls randomness in LLM outputs (0 = deterministic, 2 = very creative)

**Why it matters:**
* Low (0-0.3) for factual tasks
* Medium (0.7) for general use
* High (1.0-2.0) for creative writing

**Example:**
```
Temperature 0: "The capital of France is Paris."
Temperature 2: "Ah, the magnificent city of lights, Paris, stands as..."
```

---

### Prompt
**Definition:** The input you give to an LLM

**Why it matters:**
* Quality of prompt determines quality of output
* Well-crafted prompts = better results
* "Prompt engineering" is a skill

**Example:**
```
Bad prompt: "Write something"
Good prompt: "Write a 200-word blog intro about AI safety for non-technical readers"
```

---

## 🧠 Model Architecture

### Transformer
**Definition:** The neural network architecture that powers modern LLMs

**Why it matters:**
* Foundation of GPT, BERT, Claude, etc.
* Uses attention mechanism
* Revolutionized NLP in 2017

**Example:** All major LLMs use transformer architecture

---

### Attention / Self-Attention
**Definition:** Mechanism that lets model focus on relevant parts of input

**Why it matters:**
* Core innovation of transformers
* Allows understanding relationships between words
* "It" → can attend back to "cat" in "The cat sat. It was fluffy."

**Example:** When processing "it", model looks back at "cat"

---

### Parameters
**Definition:** The numbers (weights) that define a model

**Why it matters:**
* More parameters = more capability (usually)
* GPT-3 has 175B parameters
* Model size is often measured in parameters

**Example:** "GPT-3.5 (175B parameters)" means 175 billion numbers

---

### Fine-Tuning
**Definition:** Taking a pre-trained model and training it more on specific data

**Why it matters:**
* Much cheaper than training from scratch
* Adapts general model to specific task
* How ChatGPT was made from GPT-3

**Example:** GPT-3 (base) → + fine-tuning on conversations → ChatGPT

---

### Inference
**Definition:** Using a trained model to make predictions

**Why it matters:**
* This is what you do when you use ChatGPT
* Different from training
* Faster and cheaper than training

**Example:** Asking ChatGPT a question = inference

---

## 🎓 Training Concepts

### Pretraining
**Definition:** Initial training on massive dataset to learn language

**Why it matters:**
* Most expensive part ($millions)
* Learns world knowledge
* Creates base model

**Example:** Training GPT on internet text

---

### Loss Function
**Definition:** Measure of how wrong the model's predictions are

**Why it matters:**
* Training tries to minimize loss
* Lower loss = better model
* Different tasks use different loss functions

**Example:** If model predicts "cat" but answer is "dog", loss is high

---

### Gradient
**Definition:** Direction to adjust weights to reduce loss

**Why it matters:**
* How model learns
* Backpropagation computes gradients
* "Gradient descent" updates weights

**Example:** Like finding downhill direction on a mountain

---

### Overfitting
**Definition:** Model memorizes training data instead of learning patterns

**Why it matters:**
* Works on training data, fails on new data
* Major problem in ML
* Prevented by regularization, more data

**Example:** Memorizing answers vs understanding concepts

---

### Epoch
**Definition:** One complete pass through the training dataset

**Why it matters:**
* Training uses multiple epochs
* More epochs = more learning (but risk overfitting)

**Example:** "Trained for 10 epochs" = saw each training example 10 times

---

## 🔧 LLM Engineering

### Chain-of-Thought (CoT)
**Definition:** Prompting technique where model shows its reasoning steps

**Why it matters:**
* Improves accuracy on complex tasks
* Makes reasoning transparent
* Simple to implement

**Example:** "Let's think step-by-step: First..."

---

### RAG (Retrieval-Augmented Generation)
**Definition:** Giving LLM relevant documents to reference when answering

**Why it matters:**
* Reduces hallucination
* Adds up-to-date info
* Enables Q&A over your documents

**Example:** Search docs → Give to LLM → LLM answers based on docs

---

### Vector Database
**Definition:** Database that stores embeddings and enables similarity search

**Why it matters:**
* Powers RAG systems
* Fast similarity search
* Scales to millions of documents

**Example:** Pinecone, Weaviate, Chroma

---

### Hallucination
**Definition:** When LLM confidently generates false information

**Why it matters:**
* Major limitation of LLMs
* Can't fully prevent yet
* Must verify important outputs

**Example:** Model invents fake citations or statistics

---

### Zero-Shot / Few-Shot / Many-Shot
**Definition:** How many examples you give before asking model to perform task

**Why it matters:**
* More examples = better performance (usually)
* Shows model what you want
* Few-shot is often sweet spot

**Example:**
```
Zero-shot: "Translate to French: Hello"
Few-shot: "English: Hello, French: Bonjour
          English: Goodbye, French: Au revoir
          English: Thank you, French: ?"
```

---

### System Prompt
**Definition:** Instructions that apply to entire conversation

**Why it matters:**
* Sets model's behavior/role
* Persists across messages
* Different from user prompts

**Example:** "You are a helpful coding assistant. Always explain your code."

---

## 🤖 Advanced Concepts

### Agent
**Definition:** LLM that can use tools and take actions

**Why it matters:**
* Goes beyond just text generation
* Can search web, run code, etc.
* Future of LLM applications

**Example:** ChatGPT with plugins, AutoGPT

---

### Tool Use / Function Calling
**Definition:** LLM's ability to call external functions/APIs

**Why it matters:**
* Extends capabilities beyond text
* Can get real-time data
* Enables automation

**Example:** LLM calls calculator for math, weather API for forecast

---

### Constitutional AI
**Definition:** Training approach where model follows principles/values

**Why it matters:**
* Improves safety
* Reduces harmful outputs
* Alternative to RLHF

**Example:** Anthropic's approach with Claude

---

### RLHF (Reinforcement Learning from Human Feedback)
**Definition:** Training method where humans rank outputs, model learns preferences

**Why it matters:**
* How ChatGPT became helpful
* Aligns model with human values
* Expensive but effective

**Example:** Humans rate responses, model learns to generate higher-rated ones

---

### Quantization
**Definition:** Reducing precision of model weights to save memory

**Why it matters:**
* Runs large models on smaller hardware
* Faster inference
* Minimal quality loss

**Example:** 16-bit → 8-bit → 4-bit models

---

### LoRA (Low-Rank Adaptation)
**Definition:** Efficient fine-tuning by adding small adapter layers

**Why it matters:**
* 100x less storage than full fine-tuning
* Faster training
* Same quality

**Example:** Instead of updating 175B parameters, update 1B adapter

---

## 🎯 Model Types

### Base Model
**Definition:** Model trained only on text prediction, not instruction-following

**Why it matters:**
* Foundation for instruction models
* Not directly useful for chat
* Needs fine-tuning

**Example:** GPT-3 (base) vs ChatGPT (fine-tuned)

---

### Instruction Model
**Definition:** Model fine-tuned to follow instructions

**Why it matters:**
* What you actually use (ChatGPT, Claude, etc.)
* Understands questions/commands
* More helpful than base models

**Example:** ChatGPT, Claude, Gemini

---

### Multimodal Model
**Definition:** Model that handles multiple types of input (text, images, audio)

**Why it matters:**
* More versatile
* Can understand images + answer questions
* Future of AI

**Example:** GPT-4 Vision, Gemini

---

## 📊 Metrics

### Perplexity
**Definition:** Measure of how surprised model is by text (lower = better)

**Why it matters:**
* Common metric for language models
* Lower = model understands better
* Used to compare models

**Example:** Perplexity 20 is better than 50

---

### Top-k Sampling
**Definition:** Only consider top k most likely tokens when generating

**Why it matters:**
* Prevents very unlikely words
* Controls randomness
* Common generation strategy

**Example:** k=10 means only pick from 10 most likely next words

---

### Top-p (Nucleus) Sampling
**Definition:** Consider smallest set of tokens whose probabilities sum to p

**Why it matters:**
* Adaptive cutoff (unlike fixed top-k)
* More natural outputs
* Standard in modern LLMs

**Example:** p=0.9 means pick from tokens that cover 90% probability mass

---

## 🔗 Quick Reference

**For Understanding:**
* Token, Embedding, Context Window
* Transformer, Attention, Parameters

**For Using:**
* Prompt, Temperature, System Prompt
* Chain-of-Thought, RAG

**For Building:**
* Fine-Tuning, Inference, Vector Database
* Agent, Tool Use

**For Evaluating:**
* Hallucination, Perplexity
* Zero-shot, Few-shot

---

## 🤔 Test Your Knowledge

Can you explain these to a friend?
1. Token
2. RAG
3. Hallucination
4. Fine-Tuning
5. Agent

---

**Next:** [10 - Resources and Next Steps →](../10_RESOURCES_AND_NEXT_STEPS/README.md)
