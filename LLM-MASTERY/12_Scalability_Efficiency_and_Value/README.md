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
