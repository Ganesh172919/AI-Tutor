# Module 05: Code Understanding and Mindset Shift

## 📚 Overview

As an AI builder, you'll read thousands of lines of code—transformers, training scripts, research implementations. This module teaches you techniques to **rapidly comprehend large codebases**, shifting from linear reading to hierarchical understanding. You'll also learn to leverage LLMs as coding partners.

## 🎯 Learning Objectives

1. **Master** rapid code comprehension techniques (10x faster reading)
2. **Build** abstraction hierarchies for understanding large repos
3. **Use** LLMs effectively as coding assistants (Copilot, ChatGPT)
4. **Generate** code from natural language specifications
5. **Debug** complex issues with AI assistance
6. **Navigate** repositories at the architectural level

## 📖 Module Contents

### The Mindset Shift

**Traditional approach** (slow):
```
Read line 1 → Read line 2 → Read line 3 → ... → Read line 1000
Total time: Hours for a single file
```

**Expert approach** (fast):
```
1. Skim file structure (10 sec)
2. Identify key functions (30 sec)
3. Read critical paths (2 min)
4. Build mental model (1 min)
5. Dive into details as needed

Total time: 3-4 minutes for high-level understanding
```

### Technique 1: The Pyramid Method

**Level 1: File System Structure** (30 seconds)
```bash
tree -L 2 transformers/
```
```
transformers/
├── models/
│   ├── bert/
│   ├── gpt2/
│   ├── llama/
│   └── ...
├── trainers/
├── tokenizers/
└── utils/
```

**Insight**: Organized by model type, not by function. Models are self-contained modules.

**Level 2: Module Interfaces** (1 minute)
```python
# Read __init__.py files
from transformers import (
    BertModel,  # Main model class
    BertTokenizer,  # Tokenization
    BertConfig,  # Configuration
)
```

**Insight**: Each model exposes Model, Tokenizer, Config. Consistent API.

**Level 3: Class Structure** (2 minutes)
```python
class BertModel(PreTrainedModel):
    def __init__(self, config):  # Initialization
        ...
    
    def forward(self, input_ids, attention_mask):  # Main logic
        ...
    
    def from_pretrained(cls, path):  # Loading
        ...
```

**Insight**: `forward()` is the core. Everything else is setup/utilities.

**Level 4: Implementation Details** (as needed)
Only read detailed implementation when necessary.

### Technique 2: Follow the Data Flow

**Example: Tracing how text becomes output in GPT-2**

```python
# Input
text = "Hello, world!"

# Step 1: Tokenization
tokens = tokenizer.encode(text)
# → [15496, 11, 995, 0]

# Step 2: Embedding lookup
embeddings = model.wte(tokens)  # Word token embeddings
# → (4, 768) tensor

# Step 3: Add positional embeddings
positions = torch.arange(len(tokens))
pos_embeddings = model.wpe(positions)
hidden_states = embeddings + pos_embeddings
# → (4, 768) tensor

# Step 4: Transformer blocks (12 layers)
for block in model.h:
    hidden_states = block(hidden_states)
# → (4, 768) tensor (transformed)

# Step 5: Final layer norm
hidden_states = model.ln_f(hidden_states)
# → (4, 768) tensor (normalized)

# Step 6: Project to vocabulary
logits = model.lm_head(hidden_states)
# → (4, 50257) tensor (vocabulary size)

# Step 7: Sample next token
next_token = torch.argmax(logits[-1])
# → Single token ID
```

**Key insight**: Data flows through embeddings → transformers → projection. Simple pipeline!

### Technique 3: Code Patterns Recognition

**Pattern 1: PyTorch Module Structure**
```python
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers
        self.layer1 = nn.Linear(...)
        self.layer2 = nn.Linear(...)
    
    def forward(self, x):
        # Define computation
        x = self.layer1(x)
        x = self.layer2(x)
        return x
```
**Recognize**: Layers in `__init__`, logic in `forward`. Once you see it 10 times, skip to unique parts.

**Pattern 2: Attention Mechanism**
```python
# Always follows this structure
Q = self.q_proj(hidden_states)
K = self.k_proj(hidden_states)
V = self.v_proj(hidden_states)

scores = Q @ K.T / sqrt(d_k)
attention = softmax(scores)
output = attention @ V
```
**Recognize**: Q/K/V projection → scores → softmax → output. Skip the details, focus on modifications.

**Pattern 3: Training Loop**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward
        outputs = model(batch)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        optimizer.zero_grad()
```
**Recognize**: Standard training. Only look at loss function and optimizer for uniqueness.

### Technique 4: LLM-Assisted Code Reading

**Use Case 1: Quick Summary**
```
Prompt to GPT-4:
"Summarize this Python class in 3 bullet points:
[paste 200-line class]"

Output:
• Implements a transformer decoder with multi-head attention
• Uses rotary position embeddings (RoPE) instead of sinusoidal
• Includes KV-cache for efficient generation
```

**Use Case 2: Explain Complex Logic**
```
Prompt:
"Explain what this code does and why:
[paste confusing 20-line snippet]"
```

**Use Case 3: Find Bugs**
```
Prompt:
"Review this code for potential bugs:
[paste code]"
```

### Technique 5: Debugging with AI

**Example: Using GPT-4 to debug**

```python
# Buggy code
def attention(Q, K, V):
    scores = Q @ K.T
    attention = softmax(scores)
    return attention @ V

# Error: RuntimeError: matrix multiplication size mismatch
```

**Prompt to GPT-4**:
```
This attention function is giving a matrix multiplication error. Debug it:

def attention(Q, K, V):
    scores = Q @ K.T
    attention = softmax(scores)
    return attention @ V

Q shape: (batch, seq_len, d_k)
K shape: (batch, seq_len, d_k)
V shape: (batch, seq_len, d_v)
```

**GPT-4 Response**:
```
The issue is that Q @ K.T should have shape (batch, seq_len, seq_len), 
but you're computing it without handling the batch dimension correctly.

Fixed code:

def attention(Q, K, V):
    # Transpose K along last two dimensions
    scores = Q @ K.transpose(-2, -1)  # (batch, seq_len, seq_len)
    attention = softmax(scores, dim=-1)
    return attention @ V  # (batch, seq_len, d_v)
```

### Real-World Example: Understanding Llama 2

**Task**: Understand Llama 2 implementation in <10 minutes

**Step 1**: File structure (30 sec)
```
llama/
├── model.py          # Main model
├── tokenizer.py      # Tokenization
├── generation.py     # Text generation
└── config.json       # Configuration
```

**Step 2**: Model architecture (2 min)
```python
# In model.py
class LlamaModel(nn.Module):
    def __init__(self, config):
        self.layers = [LlamaDecoderLayer(...) for _ in range(config.num_layers)]
        self.norm = RMSNorm(...)
    
    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

**Key insights**:
- Decoder-only (no encoder)
- RMSNorm instead of LayerNorm
- Standard transformer stack

**Step 3**: Unique features (3 min)
```python
# RoPE (Rotary Position Embeddings)
def apply_rotary_pos_emb(q, k, freqs_cis):
    # Rotate queries and keys
    q = q * freqs_cis
    k = k * freqs_cis
    return q, k

# SwiGLU activation (instead of ReLU)
def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)
```

**Step 4**: Generation (2 min)
```python
# Top-k, top-p sampling
def sample(logits, temperature=0.7, top_k=50, top_p=0.9):
    # Temperature scaling
    logits = logits / temperature
    
    # Top-k filtering
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    
    # Top-p (nucleus) sampling
    probs = softmax(top_k_logits)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    mask = cumulative_probs < top_p
    
    # Sample
    token = torch.multinomial(probs[mask], 1)
    return token
```

**Total time**: ~8 minutes
**Understanding**: 80% of how Llama works

## 💻 Tools and Practices

### GitHub Copilot Best Practices

```python
# Tip 1: Write descriptive comments
# Generate a function that implements scaled dot-product attention
# It should take Q, K, V tensors and return output + attention weights
def attention(Q, K, V):
    # Copilot generates the implementation!
    pass

# Tip 2: Use type hints
def multi_head_attention(
    query: torch.Tensor,  # (batch, seq_len, d_model)
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Copilot understands the types
    pass

# Tip 3: Start with tests
def test_attention():
    Q = torch.randn(2, 10, 64)
    K = torch.randn(2, 10, 64)
    V = torch.randn(2, 10, 64)
    
    output, weights = attention(Q, K, V)
    
    assert output.shape == (2, 10, 64)
    assert weights.shape == (2, 10, 10)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 10))
```

### ChatGPT Code Generation

**Prompt engineering for code**:

❌ **Bad prompt**:
```
Write transformer code
```

✅ **Good prompt**:
```
Write a PyTorch implementation of a transformer encoder layer with:
- Multi-head self-attention (8 heads)
- Feed-forward network with GELU activation
- Layer normalization (pre-norm architecture)
- Residual connections
- Dropout (0.1)

Input: (batch_size, seq_len, d_model=512)
Output: same shape

Include docstrings and type hints.
```

## ✏️ Exercises

1. **Speed Reading**: Read the Llama 2 codebase in <15 minutes, create architecture diagram
2. **Code Generation**: Use Copilot to implement a complete transformer from spec
3. **Debugging Challenge**: Fix 10 buggy code snippets with AI assistance
4. **Repository Analysis**: Analyze a new repo (e.g., Mistral) and document key differences from GPT-2

## 🎯 Key Takeaways

1. **Don't read linearly**: Start with structure, dive into details selectively
2. **Follow data flow**: Trace inputs → outputs to understand logic
3. **Recognize patterns**: Skip boilerplate, focus on unique implementations
4. **Use AI assistants**: Copilot for generation, ChatGPT for explanation
5. **Practice daily**: Code reading is a skill that improves with practice

**After this module, you can understand a new LLM implementation in <30 minutes instead of days.**

---

**Next**: Module 06 - Human Brain Inspiration and Control Mechanisms →
