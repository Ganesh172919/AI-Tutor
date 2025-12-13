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
