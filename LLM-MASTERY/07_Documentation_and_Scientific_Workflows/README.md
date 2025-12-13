# Module 07: Documentation and Scientific Workflows

Documentation is the bridge between your brilliant ideas and their impact on the world. This module teaches research-grade documentation, experiment tracking, and scientific workflows.

## Key Topics

### 1. Research Documentation with LaTeX
- Writing technical papers
- Mathematical notation
- Bibliography management (BibTeX)
- Templates: NeurIPS, ICML, ACL

### 2. Experiment Tracking
**Weights & Biases (W&B)**:
```python
import wandb

wandb.init(project="llm-training", config={
    "learning_rate": 5e-5,
    "batch_size": 32,
    "model": "gpt2-medium"
})

for step in range(1000):
    loss = train_step()
    wandb.log({"loss": loss, "step": step})
```

### 3. Reproducibility Best Practices
- Random seed management
- Docker containers for environments
- Data versioning (DVC)
- Model checkpointing

### 4. Technical Writing
- Blog posts for practitioners
- Tutorial writing
- README best practices
- Documentation for open-source

---

# Module 08: Real-World Applications

## Case Study 1: ChatGPT Architecture

### System Components:
1. **Base Model**: GPT-3.5/GPT-4 (175B-1.7T parameters)
2. **RLHF Pipeline**: Supervised fine-tuning → Reward model → PPO
3. **Moderation**: Toxicity filters, prompt injection detection
4. **Serving**: Distributed inference across 1000+ GPUs
5. **Caching**: Common queries cached for speed
6. **Rate Limiting**: Prevent abuse
7. **Monitoring**: Real-time performance tracking

### Lessons Learned:
- RLHF dramatically improves user satisfaction
- Moderation is critical for safety
- Caching reduces costs by 40%
- User feedback loops enable continuous improvement

## Case Study 2: GitHub Copilot

### Architecture:
- **Model**: Codex (GPT-3 fine-tuned on code)
- **Context**: IDE integration captures file context
- **Latency**: <100ms for suggestions
- **Privacy**: No code sent to logs without permission

### Engineering Challenges:
- Real-time inference at scale
- Context-aware suggestions
- Multi-language support
- Balancing suggestion quality vs. latency

## Case Study 3: Production Deployment

### FastAPI Serving:
```python
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
generator = pipeline("text-generation", model="gpt2")

@app.post("/generate")
async def generate(prompt: str, max_length: int = 100):
    result = generator(prompt, max_length=max_length)[0]
    return {"generated_text": result['generated_text']}
```

### Monitoring with Prometheus:
```python
from prometheus_client import Counter, Histogram

request_count = Counter('llm_requests_total', 'Total requests')
latency = Histogram('llm_latency_seconds', 'Request latency')

@app.post("/generate")
@latency.time()
async def generate(prompt: str):
    request_count.inc()
    # ... generation code
```

---

# Module 10: Skill Building and Exercises

## Progressive Coding Challenges

### Lab 1: Tokenizer Implementation
```python
# Challenge: Build a BPE tokenizer from scratch
# Requirements:
# - Train on corpus of 10K sentences
# - Vocabulary size of 1000 tokens
# - Implement encode() and decode()
# - Handle unknown tokens gracefully

# Starter code
class BPETokenizer:
    def train(self, corpus: List[str], vocab_size: int):
        # TODO: Implement BPE training
        pass
    
    def encode(self, text: str) -> List[int]:
        # TODO: Implement encoding
        pass
    
    def decode(self, token_ids: List[int]) -> str:
        # TODO: Implement decoding
        pass
```

### Lab 2: Attention Visualization
```python
# Challenge: Visualize attention patterns
# Requirements:
# - Load pre-trained transformer
# - Extract attention weights from all layers
# - Create heatmap visualization
# - Identify most-attended tokens

# Expected output: 12 heatmaps (one per layer) showing attention patterns
```

### Lab 3: Fine-Tuning on Custom Data
```python
# Challenge: Fine-tune GPT-2 on your own dataset
# Requirements:
# - Collect 10K examples in your domain
# - Clean and preprocess data
# - Fine-tune with LoRA (< 1% parameters)
# - Evaluate on held-out test set
# - Achieve <2.5 perplexity

# Bonus: Deploy as API endpoint
```

### Lab 4: Chain-of-Thought Reasoning
```python
# Challenge: Build CoT reasoning system
# Requirements:
# - Implement zero-shot CoT ("Let's think step by step")
# - Test on GSM8K math problems
# - Compare with direct answering
# - Achieve >70% accuracy

# Example:
# Problem: "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many now?"
# CoT: "Roger starts with 5. He buys 2 cans × 3 balls = 6. Total = 5 + 6 = 11"
# Answer: 11 ✓
```

### Lab 5: RAG System
```python
# Challenge: Build Retrieval-Augmented Generation
# Requirements:
# - Index 1000+ documents with FAISS
# - Implement semantic search
# - Generate answers with citations
# - Compare RAG vs. non-RAG accuracy

# Example:
# Query: "What is the capital of France?"
# Retrieved: "Paris is the capital and largest city of France..."
# Generated: "The capital of France is Paris [1]"
```

## Capstone Project

**Build a Domain-Specific LLM Application**

### Example: Medical Diagnosis Assistant
1. **Data Collection**: Gather medical literature, case studies
2. **Model Selection**: Start with Llama-2-7B
3. **Fine-Tuning**: Train on medical Q&A with LoRA
4. **Reasoning**: Implement CoT for diagnosis
5. **RAG**: Integrate medical knowledge base
6. **Safety**: Add disclaimer, cite sources
7. **Deployment**: FastAPI + React frontend
8. **Evaluation**: Test with medical professionals

**Deliverables**:
- ✅ Working application
- ✅ Documentation (README, API docs)
- ✅ Evaluation report
- ✅ Presentation (10 min)

---

# Module 12: Scalability, Efficiency, and Value

## Quantization Deep Dive

### 8-bit Quantization
```python
from transformers import AutoModelForCausalLM
import torch

# Load model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Memory usage: ~7GB instead of ~26GB!
# Speed: ~2x faster
# Accuracy: <1% degradation
```

### GPTQ (Advanced Quantization)
```python
from auto_gptq import AutoGPTQForCausalLM

# 4-bit quantization with GPTQ
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-13B-GPTQ",
    use_safetensors=True,
    device="cuda:0"
)

# Memory: ~3.5GB (4x reduction!)
# Speed: 3-4x faster than fp16
# Accuracy: ~2% degradation
```

## Knowledge Distillation

### Teacher-Student Training
```python
class Distiller:
    def __init__(self, teacher, student, temperature=2.0):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
    
    def distill_step(self, batch):
        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(batch["input_ids"])
        
        # Get student predictions
        student_logits = self.student(batch["input_ids"])
        
        # Distillation loss (soft targets)
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Task loss (hard targets)
        task_loss = F.cross_entropy(student_logits, batch["labels"])
        
        # Combined loss
        loss = 0.5 * distill_loss + 0.5 * task_loss
        return loss

# Result: 70B teacher → 7B student retains 95% performance!
```

## Edge Deployment

### Mobile Inference with llama.cpp
```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert model to GGML format (4-bit quantization)
python convert.py /path/to/llama-7b --outtype q4_0

# Run on CPU (even Raspberry Pi!)
./main -m models/llama-7b-q4_0.bin -p "Hello, world!" -n 128

# Performance on iPhone 14 Pro:
# - Model: Llama-7B (4-bit)
# - Speed: 15 tokens/second
# - Memory: 4GB
```

## Production Metrics

### Latency Optimization
```python
# Techniques for faster inference
1. Batch processing (10x throughput)
2. KV-cache (2x speedup for generation)
3. Quantization (2-4x faster)
4. Flash Attention (2x faster)
5. Speculative decoding (2-3x faster)

# Combined: 100x improvement possible!
```

### Cost Analysis
```python
# GPT-4 API vs. Self-Hosted Llama-70B

# GPT-4 API:
# - Cost: $15-60 per 1M tokens
# - Latency: ~1-2 seconds
# - Maintenance: Zero
# - Scale: Infinite

# Self-Hosted Llama-70B (8-bit):
# - Hardware: 4x A100 GPUs ($40K or $8/hour on cloud)
# - Throughput: 100 tokens/sec = 8.6M tokens/day
# - Cost per 1M tokens: ~$2 (amortized)
# - Setup: High (infrastructure, DevOps)
# - Break-even: ~10M tokens/day

# Recommendation:
# - < 1M tokens/day: Use API
# - 1M-10M tokens/day: Consider self-hosting
# - > 10M tokens/day: Definitely self-host
```

## The End-to-End LLM Builder

**You now know**:
- ✅ How to build transformers from scratch
- ✅ How to train on billions of tokens
- ✅ How to fine-tune for any task
- ✅ How to add reasoning capabilities
- ✅ How to deploy at production scale
- ✅ How to optimize for cost and speed
- ✅ How to build next-gen AI systems

**Go build the future!** 🚀
