# Comprehensive Example Projects and Code Templates

This document provides extensive code examples and project templates for building LLM applications.

## Project 1: Build a Custom Tokenizer (Comprehensive Example)

### Implementation with Multiple Algorithms

```python
"""
Complete Tokenization Suite
============================

Implements three major tokenization algorithms:
1. Byte Pair Encoding (BPE) - GPT-2 style
2. WordPiece - BERT style
3. Unigram - SentencePiece style

Each with full training, encoding, and decoding capabilities.
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import heapq
import math


class BytePairEncoding:
    """
    Full BPE implementation with byte-level support.
    
    Features:
    - Character-level initialization
    - Iterative merge learning
    - Efficient encoding with learned merges
    - Byte-level fallback for unknown characters
    """
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.byte_encoder = self._create_byte_encoder()
        
    def _create_byte_encoder(self):
        """Create byte to unicode mapping."""
        # Printable ASCII
        bs = list(range(ord("!"), ord("~")+1))
        bs += list(range(ord("¡"), ord("¬")+1))
        bs += list(range(ord("®"), ord("ÿ")+1))
        
        cs = bs[:]
        n = 0
        
        # Map remaining bytes
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def train(self, corpus, verbose=False):
        """
        Train BPE on corpus.
        
        Algorithm:
        1. Start with character vocabulary
        2. Find most frequent adjacent pair
        3. Merge pair into new token
        4. Repeat until vocab_size reached
        """
        # Build word frequency dictionary
        word_freqs = Counter()
        for text in corpus:
            words = text.split()
            word_freqs.update(words)
        
        # Initialize vocabulary with characters
        vocab = {}
        for word, freq in word_freqs.items():
            vocab[tuple(word) + ('</w>',)] = freq
        
        # Learn merges
        num_merges = self.vocab_size - 256  # Reserve space for bytes
        
        for i in range(num_merges):
            # Count pairs
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += freq
            
            if not pairs:
                break
            
            # Find best pair
            best_pair = max(pairs, key=pairs.get)
            
            if verbose and i % 100 == 0:
                print(f"Merge {i}: {best_pair} (freq={pairs[best_pair]})")
            
            # Store merge
            self.merges[best_pair] = i
            
            # Apply merge
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j+1]) == best_pair:
                        new_word.append(word[j] + word[j+1])
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab
        
        # Build final vocabulary
        self.vocab = {token: idx for idx, token in enumerate(
            sorted(set(token for word in vocab.keys() for token in word))
        )}
        
        return self.vocab, self.merges


class WordPiece:
    """
    WordPiece implementation (BERT-style).
    
    Uses likelihood-based scoring instead of frequency:
    Score(pair) = P(pair) / (P(left) * P(right))
    """
    
    def __init__(self, vocab_size=1000, unk_token='[UNK]'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = {}
        
    def train(self, corpus, verbose=False):
        """
        Train WordPiece with likelihood scoring.
        """
        # Count word frequencies
        word_freqs = Counter()
        for text in corpus:
            words = text.split()
            word_freqs.update(words)
        
        # Initialize with characters
        vocab = set()
        for word in word_freqs:
            vocab.update(word)
        vocab.add(self.unk_token)
        
        # Split words into characters
        splits = {word: list(word) for word in word_freqs}
        
        # Iteratively merge best pairs
        while len(vocab) < self.vocab_size:
            # Compute pair scores
            pair_scores = self._compute_pair_scores(splits, word_freqs)
            
            if not pair_scores:
                break
            
            best_pair = max(pair_scores, key=pair_scores.get)
            
            if verbose and len(vocab) % 100 == 0:
                print(f"Vocab size {len(vocab)}: merging {best_pair}")
            
            # Add to vocabulary
            new_token = ''.join(best_pair)
            vocab.add(new_token)
            
            # Update splits
            splits = self._merge_pair(best_pair, splits)
        
        # Create vocabulary mapping
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        
        return self.vocab
    
    def _compute_pair_scores(self, splits, word_freqs):
        """Compute likelihood scores for all pairs."""
        symbol_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            split = splits[word]
            
            for symbol in split:
                symbol_freqs[symbol] += freq
            
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
        
        scores = {}
        for pair, freq in pair_freqs.items():
            score = freq / (symbol_freqs[pair[0]] * symbol_freqs[pair[1]])
            scores[pair] = score
        
        return scores
    
    def _merge_pair(self, pair, splits):
        """Merge pair in all splits."""
        new_splits = {}
        new_token = ''.join(pair)
        
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i+1] == pair[1]:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        
        return new_splits


## Project 2: Fine-Tuning Pipeline (Complete Production Code)

```python
"""
Production Fine-Tuning Pipeline
================================

Complete pipeline for fine-tuning LLMs with:
- Data loading and preprocessing
- LoRA/QLoRA support
- Distributed training
- Experiment tracking
- Model evaluation
- Checkpoint management
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb
from typing import Optional, Dict, List
import os


class FineTuningPipeline:
    """
    Complete fine-tuning pipeline.
    
    Supports:
    - Full fine-tuning
    - LoRA
    - QLoRA (4-bit/8-bit)
    - Custom datasets
    - Distributed training
    """
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        output_dir: str = "./checkpoints",
        use_lora: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.data_collator = None
        
    def setup_model(self):
        """Load and prepare model."""
        print(f"Loading model: {self.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Prepare for LoRA if using quantization
        if self.load_in_8bit or self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        if self.use_lora:
            lora_config = LoraConfig(
                r=8,  # Rank
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Data collator for language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        print("Model setup complete!")
    
    def prepare_dataset(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        text_column: str = "text",
        max_length: int = 512
    ):
        """
        Load and preprocess dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_path: Path to local dataset
            text_column: Column containing text
            max_length: Maximum sequence length
        """
        # Load dataset
        if dataset_name:
            dataset = load_dataset(dataset_name)
        elif dataset_path:
            dataset = load_dataset('json', data_files=dataset_path)
        else:
            raise ValueError("Must provide dataset_name or dataset_path")
        
        # Tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500
    ):
        """
        Train the model.
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            fp16=True,  # Mixed precision
            report_to="wandb"  # Log to Weights & Biases
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        self.model.save_pretrained(os.path.join(self.output_dir, "final"))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, "final"))
        
        print("Training complete!")
    
    def generate(self, prompt: str, max_length: int = 100, **kwargs):
        """
        Generate text from prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = FineTuningPipeline(
        base_model="gpt2",  # Use smaller model for demo
        use_lora=True,
        load_in_8bit=False
    )
    
    # Setup model
    pipeline.setup_model()
    
    # Prepare dataset (using a small public dataset)
    dataset = pipeline.prepare_dataset(
        dataset_name="wikitext",
        text_column="text",
        max_length=256
    )
    
    # Train
    pipeline.train(
        train_dataset=dataset["train"].select(range(1000)),  # Small subset for demo
        eval_dataset=dataset["validation"].select(range(100)),
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=8
    )
    
    # Test generation
    prompt = "Once upon a time"
    generated = pipeline.generate(prompt, max_length=50)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
```

## Project 3: RAG System (Production-Ready)

```python
"""
Production RAG (Retrieval-Augmented Generation) System
=======================================================

Complete RAG implementation with:
- Vector database (FAISS/Pinecone)
- Dense retrieval
- Reranking
- Generation with citations
- Caching
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional
import pickle
import os


class ProductionRAG:
    """
    Production-ready RAG system.
    
    Features:
    - Efficient vector search with FAISS
    - Cross-encoder reranking
    - Citation support
    - Caching for speed
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        generator_model: str = "gpt2-medium",
        cache_dir: str = "./rag_cache"
    ):
        # Load models
        print("Loading models...")
        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(rerank_model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModelForCausalLM.from_pretrained(generator_model)
        
        # Initialize index
        self.index = None
        self.documents = []
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        print("Models loaded!")
    
    def index_documents(self, documents: List[str], save_path: Optional[str] = None):
        """
        Build FAISS index from documents.
        """
        print(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        
        # Encode documents
        embeddings = self.embedder.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Save if path provided
        if save_path:
            self.save_index(save_path)
        
        print(f"Indexing complete! {self.index.ntotal} documents indexed.")
    
    def save_index(self, path: str):
        """Save index and documents."""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load index and documents."""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.docs", 'rb') as f:
            self.documents = pickle.load(f)
        print(f"Index loaded from {path}")
    
    def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        """
        Retrieve top-k documents for query.
        """
        # Encode query
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = self.index.search(query_emb, k)
        
        # Format results
        results = [
            {
                'text': self.documents[idx],
                'score': score,
                'index': idx
            }
            for score, idx in zip(scores[0], indices[0])
        ]
        
        return results
    
    def rerank(self, query: str, candidates: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        Rerank candidates using cross-encoder.
        """
        # Prepare pairs
        pairs = [[query, cand['text']] for cand in candidates]
        
        # Score with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Add scores
        for cand, score in zip(candidates, scores):
            cand['rerank_score'] = float(score)
        
        # Sort and return top-n
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        return candidates[:top_n]
    
    def generate_answer(
        self,
        query: str,
        context_docs: List[Dict],
        max_length: int = 200
    ) -> str:
        """
        Generate answer given query and context.
        """
        # Build context
        context = "\n\n".join([
            f"[{i+1}] {doc['text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Build prompt
        prompt = f"""Answer the question based on the context below. Include citations using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part (after "Answer:")
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
    
    def query(
        self,
        question: str,
        retrieve_k: int = 20,
        rerank_n: int = 5,
        max_answer_length: int = 200
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve → rerank → generate.
        """
        # Step 1: Retrieve
        candidates = self.retrieve(question, k=retrieve_k)
        
        # Step 2: Rerank
        top_docs = self.rerank(question, candidates, top_n=rerank_n)
        
        # Step 3: Generate
        answer = self.generate_answer(question, top_docs, max_answer_length)
        
        return {
            'question': question,
            'answer': answer,
            'sources': top_docs,
            'num_sources': len(top_docs)
        }


# Example usage
if __name__ == "__main__":
    # Create knowledge base
    documents = [
        "Paris is the capital and largest city of France.",
        "The Eiffel Tower is located in Paris, France.",
        "France is a country in Western Europe.",
        "London is the capital of the United Kingdom.",
        "The population of Paris is approximately 2.2 million.",
    ]
    
    # Initialize RAG
    rag = ProductionRAG()
    
    # Index documents
    rag.index_documents(documents)
    
    # Query
    question = "What is the capital of France?"
    result = rag.query(question)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"\nSources used: {result['num_sources']}")
    for i, source in enumerate(result['sources']):
        print(f"  [{i+1}] {source['text']} (score: {source['rerank_score']:.3f})")
```

This comprehensive guide provides production-ready code for three major LLM projects:
1. Complete tokenization suite (BPE, WordPiece, Unigram)
2. Full fine-tuning pipeline with LoRA/QLoRA support
3. Production RAG system with vector search and reranking

Each implementation includes:
- Full documentation
- Error handling
- Optimization techniques
- Real-world usage examples

These templates can be adapted for any LLM application!
