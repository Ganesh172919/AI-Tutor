# Module 01: Fundamentals of Large Language Models

## 📚 Overview

Welcome to the foundational module of LLM Mastery! This module establishes the core concepts, mathematical foundations, and practical implementations you need to understand how Large Language Models work at a deep level.

## 🎯 Learning Objectives

By the end of this module, you will:

1. **Understand** the evolution from traditional NLP to modern LLMs
2. **Implement** tokenization algorithms (BPE, WordPiece, SentencePiece) from scratch
3. **Master** embedding representations and their mathematical properties
4. **Build** attention mechanisms in NumPy and PyTorch
5. **Explain** why attention is the breakthrough that enabled modern LLMs
6. **Apply** these concepts to real-world text processing tasks

## 📖 Module Contents

### 01. What are LLMs? (`01_what_are_llms.md`)
- Evolution from bag-of-words to transformers
- The transformer revolution (2017-2025)
- Key milestones: BERT, GPT-3, ChatGPT, GPT-4, Claude, Llama
- Scaling laws and emergent abilities
- **Code**: Timeline visualization with Python

### 02. Tokenization (`02_tokenization.md`)
- Why we need tokenization
- Byte Pair Encoding (BPE) algorithm walkthrough
- WordPiece (BERT's tokenizer)
- SentencePiece (language-agnostic)
- Handling rare words, punctuation, multilingual text
- **Code**: Complete BPE, WordPiece, SentencePiece implementations

### 03. Embeddings (`03_embeddings.md`)
- From one-hot vectors to dense embeddings
- Word2Vec (Skip-gram, CBOW) explained
- GloVe and matrix factorization
- Learned embeddings in transformers
- Positional information and why it matters
- **Code**: Train Word2Vec from scratch, visualize embeddings

### 04. Attention Mechanisms (`04_attention_mechanisms.md`)
- The attention intuition: "Spotlighting relevant words"
- Scaled dot-product attention (the formula explained)
- Self-attention vs. cross-attention
- Multi-head attention: Why multiple heads?
- Attention visualization and interpretation
- **Code**: Attention from scratch in NumPy, then PyTorch

## 💻 Code Files

| File | Description | Lines | Difficulty |
|------|-------------|-------|------------|
| `tokenizer_bpe.py` | BPE tokenizer from scratch | ~200 | ⭐⭐☆☆☆ |
| `tokenizer_wordpiece.py` | WordPiece implementation | ~250 | ⭐⭐⭐☆☆ |
| `attention_from_scratch.py` | Complete attention mechanisms | ~300 | ⭐⭐⭐⭐☆ |
| `word2vec_scratch.py` | Word2Vec training | ~400 | ⭐⭐⭐☆☆ |

## ✏️ Exercises

See `exercises.md` for:
- 10 conceptual questions with detailed answers
- 5 coding challenges (tokenize Shakespeare, build attention visualizer)
- 2 mini-projects (sentiment classifier, document similarity)

## ⏱️ Estimated Time

- **Reading**: 6-8 hours
- **Coding**: 6-8 hours
- **Exercises**: 2-3 hours
- **Total**: 14-19 hours

## 🔗 Dependencies

```bash
pip install numpy scipy matplotlib seaborn
pip install torch torchvision
pip install tokenizers transformers
pip install jupyter notebook
pip install plotly  # for interactive visualizations
```

## 📚 Required Reading

Before starting this module:
1. "Attention Is All You Need" (Vaswani et al., 2017) - Sections 1-3
2. "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
3. Hugging Face Tokenizers documentation

## 🚀 Quick Start

```python
# Run this in Jupyter or Python REPL
import sys
sys.path.append('01_Fundamentals_of_LLMs')

# Test BPE tokenizer
from tokenizer_bpe import BPETokenizer
tokenizer = BPETokenizer()
tokenizer.train(["Hello world", "Hello there", "World peace"])
tokens = tokenizer.encode("Hello world peace")
print(tokens)  # See BPE in action!

# Test attention
from attention_from_scratch import scaled_dot_product_attention
import torch
Q = torch.randn(1, 8, 64)  # Query
K = torch.randn(1, 8, 64)  # Key
V = torch.randn(1, 8, 64)  # Value
output, attention_weights = scaled_dot_product_attention(Q, K, V)
print(attention_weights.shape)  # (1, 8, 8) - attention matrix
```

## 🎯 Key Takeaways

After completing this module, you should be able to explain:

1. **Why transformers replaced RNNs/LSTMs**: Parallelization, long-range dependencies
2. **How tokenization affects model performance**: Vocabulary size, rare words, efficiency
3. **What attention does**: Dynamically weights input based on relevance
4. **Why position matters**: Transformers are permutation-invariant without positional encoding
5. **How embeddings capture meaning**: Geometric relationships in vector space

## 🔜 Next Steps

Once you've mastered this module, proceed to:
- **Module 02**: Internal Mechanics - Deep dive into transformer architecture
- **Module 10**: Build your first transformer from these components

---

**Let's build a solid foundation! Start with `01_what_are_llms.md` →**
