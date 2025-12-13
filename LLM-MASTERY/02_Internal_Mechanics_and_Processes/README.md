# Module 02: Internal Mechanics and Processes

## 📚 Overview

Now that you understand the fundamentals (tokenization, embeddings, attention), it's time to dive deep into the **transformer architecture**—the engine powering all modern LLMs from BERT to GPT-4.

This module dissects transformers layer by layer, revealing how billions of parameters work together to understand and generate human-like text.

## 🎯 Learning Objectives

By the end of this module, you will:

1. **Understand** the complete transformer architecture (encoder, decoder, encoder-decoder)
2. **Implement** multi-head attention from scratch in PyTorch
3. **Master** positional encodings (sinusoidal, learned, RoPE, ALiBi)
4. **Build** feed-forward networks and understand their role
5. **Explain** layer normalization and why it's critical for deep networks
6. **Analyze** training dynamics and scaling laws
7. **Construct** a complete transformer model from components

## 📖 Module Contents

### 01. Transformer Architecture (`01_transformer_architecture.md`)
- The original transformer (Vaswani et al., 2017)
- Encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)
- Information flow through layers
- Residual connections and layer norm
- **Code**: Complete transformer implementation

### 02. Multi-Head Attention (`02_multi_head_attention.md`)
- Why multiple heads? Different attention patterns
- Linear projections for Q, K, V
- Parallel computation across heads
- Concatenation and output projection
- **Code**: Multi-head attention with visualization

### 03. Positional Encodings (`03_positional_encodings.md`)
- Why position matters in transformers
- Sinusoidal encodings (original approach)
- Learned positional embeddings
- Rotary Position Embeddings (RoPE) - used in Llama
- ALiBi (Attention with Linear Biases)
- **Code**: All positional encoding variants

### 04. Feed-Forward Networks (`04_feedforward_networks.md`)
- The FFN structure: Linear → Activation → Linear
- Why we need FFN in addition to attention
- Activation functions (ReLU, GELU, SwiGLU)
- Dimension expansion (typically 4x model dimension)
- **Code**: FFN implementations and ablation studies

### 05. Layer Normalization (`05_layer_normalization.md`)
- Batch norm vs. layer norm vs. RMS norm
- Pre-norm vs. post-norm architectures
- Why layer norm enables deep transformers
- Gradient flow analysis
- **Code**: Normalization variants and comparisons

### 06. Training Dynamics (`06_training_dynamics.md`)
- Loss curves and convergence
- Scaling laws (Kaplan, Chinchilla)
- Learning rate schedules (warmup, cosine decay)
- Gradient accumulation for large batches
- Mixed precision training (fp16, bf16)
- **Code**: Training loop with all optimizations

## 💻 Code Files

| File | Description | Lines | Difficulty |
|------|-------------|-------|------------|
| `transformer_full.py` | Complete transformer model | ~600 | ⭐⭐⭐⭐⭐ |
| `positional_encoding.py` | All position encoding types | ~300 | ⭐⭐⭐☆☆ |
| `layer_norm_comparison.py` | Normalization experiments | ~200 | ⭐⭐⭐☆☆ |
| `training_loop.py` | Production training setup | ~400 | ⭐⭐⭐⭐☆ |

## ✏️ Exercises

1. **Implement**: Build a transformer encoder from scratch
2. **Experiment**: Compare different positional encodings on sequence tasks
3. **Analyze**: Visualize attention patterns across layers
4. **Optimize**: Benchmark different normalization strategies
5. **Train**: Train a small transformer on language modeling

## ⏱️ Estimated Time

- **Reading**: 8-10 hours
- **Coding**: 10-12 hours
- **Exercises**: 4-6 hours
- **Total**: 22-28 hours

## 🔗 Dependencies

```bash
pip install torch torchvision
pip install einops  # For tensor operations
pip install wandb  # For experiment tracking
pip install matplotlib seaborn
```

## 📚 Required Reading

1. "Attention Is All You Need" (Vaswani et al., 2017) - **Complete paper**
2. "Layer Normalization" (Ba et al., 2016)
3. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
4. "Train Short, Test Long: Attention with Linear Biases (ALiBi)" (Press et al., 2021)

## 🚀 Quick Start

```python
# Build a mini-transformer
from transformer_full import Transformer

model = Transformer(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1
)

# Forward pass
import torch
src = torch.randint(0, 10000, (32, 512))  # (batch, seq_len)
tgt = torch.randint(0, 10000, (32, 512))

output = model(src, tgt)
print(f"Output shape: {output.shape}")  # (32, 512, 10000)
```

## 🎯 Key Takeaways

After this module, you'll understand:

1. **Architecture**: How transformers process sequences in parallel
2. **Attention**: Why multi-head attention captures different patterns
3. **Position**: How models learn sequential order without recurrence
4. **Depth**: How residuals and normalization enable 96+ layer models
5. **Scale**: How computational efficiency enables billion-parameter models

## 🔜 Next Module

**Module 03**: Building Data-Driven Learning Models - Learn to train your own transformers on custom data

---

**Start with `01_transformer_architecture.md` to dive deep! →**
