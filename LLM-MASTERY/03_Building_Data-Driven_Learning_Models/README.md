# Module 03: Building Data-Driven Learning Models

## 📚 Overview

Data is the fuel for LLMs. This module teaches you how to collect, clean, augment, and use data to train custom models. You'll learn the complete pipeline from raw text to a production-ready dataset, plus fine-tuning techniques (LoRA, QLoRA, full fine-tuning) to adapt pre-trained models to your domain.

## 🎯 Learning Objectives

1. **Master** data collection from web, books, code repositories
2. **Build** data cleaning pipelines (deduplication, quality filtering)
3. **Implement** data augmentation techniques for LLMs
4. **Design** tokenization strategies for multilingual models
5. **Train** models with distributed data parallelism
6. **Fine-tune** using LoRA, QLoRA, and full parameter updates
7. **Optimize** training loops with mixed precision and gradient accumulation

## 📖 Module Contents

### 01. Data Collection (`01_data_collection.md`)

**Sources of Training Data**:

1. **Web Scraping** (Common Crawl, custom crawlers)
2. **Books** (Project Gutenberg, Books3)
3. **Code** (GitHub, Stack Overflow)
4. **Scientific Papers** (arXiv, PubMed)
5. **Dialogue** (Reddit, Twitter, forums)

**Example: Web Scraping Pipeline**

```python
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from collections import deque

class WebCrawler:
    def __init__(self, seed_urls, max_pages=1000):
        self.visited = set()
        self.to_visit = deque(seed_urls)
        self.max_pages = max_pages
        self.texts = []
        
    def is_valid_url(self, url):
        """Check if URL is valid for crawling."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)
    
    def get_page_text(self, url):
        """Extract text from a webpage."""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None
    
    def get_links(self, url, html):
        """Extract links from page."""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(url, href)
            
            if self.is_valid_url(full_url) and full_url not in self.visited:
                links.append(full_url)
        
        return links
    
    def crawl(self):
        """Main crawling loop."""
        while self.to_visit and len(self.visited) < self.max_pages:
            url = self.to_visit.popleft()
            
            if url in self.visited:
                continue
            
            print(f"Crawling ({len(self.visited)}/{self.max_pages}): {url}")
            
            text = self.get_page_text(url)
            
            if text:
                self.texts.append({
                    'url': url,
                    'text': text
                })
                
                # Extract links for future crawling
                try:
                    response = requests.get(url, timeout=10)
                    links = self.get_links(url, response.content)
                    self.to_visit.extend(links)
                except:
                    pass
            
            self.visited.add(url)
            time.sleep(1)  # Be polite
        
        return self.texts

# Example usage
crawler = WebCrawler(['https://en.wikipedia.org/wiki/Machine_learning'], max_pages=100)
data = crawler.crawl()
print(f"Collected {len(data)} pages")
```

### 02. Data Cleaning (`02_data_cleaning.md`)

**The Data Cleaning Pipeline**:

```
Raw Data (billions of tokens)
    ↓
1. Deduplication (remove exact and near-duplicates)
    ↓
2. Quality Filtering (language detection, perplexity scoring)
    ↓
3. PII Removal (personal identifiable information)
    ↓
4. Toxic Content Filtering (hate speech, explicit content)
    ↓
5. Format Normalization (consistent encoding, whitespace)
    ↓
Clean Data (ready for training)
```

**Deduplication with MinHash**:

```python
from datasketch import MinHash, MinHashLSH
import re

class DocumentDeduplicator:
    def __init__(self, threshold=0.8, num_perm=128):
        """
        Initialize deduplicator.
        
        Args:
            threshold: Jaccard similarity threshold (0-1)
            num_perm: Number of permutations for MinHash
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
    def preprocess(self, text):
        """Convert text to tokens for hashing."""
        # Lowercase and split into words
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def create_minhash(self, tokens):
        """Create MinHash signature from tokens."""
        m = MinHash(num_perm=self.num_perm)
        for token in tokens:
            m.update(token.encode('utf8'))
        return m
    
    def deduplicate(self, documents):
        """
        Remove near-duplicate documents.
        
        Args:
            documents: List of text documents
        
        Returns:
            List of unique documents
        """
        unique_docs = []
        
        for i, doc in enumerate(documents):
            # Preprocess and hash
            tokens = self.preprocess(doc)
            minhash = self.create_minhash(tokens)
            
            # Check for duplicates
            result = self.lsh.query(minhash)
            
            if len(result) == 0:
                # No duplicates found, this is unique
                self.lsh.insert(f"doc_{i}", minhash)
                unique_docs.append(doc)
            else:
                print(f"Document {i} is similar to {result[0]}, skipping")
        
        return unique_docs

# Example
deduplicator = DocumentDeduplicator(threshold=0.8)
docs = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumps over the lazy dog",  # Exact duplicate
    "The fast brown fox leaps over the lazy dog",   # Near-duplicate
    "Completely different content here"
]

unique = deduplicator.deduplicate(docs)
print(f"Original: {len(docs)} docs, After dedup: {len(unique)} docs")
```

**Quality Filtering**:

```python
import langdetect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QualityFilter:
    def __init__(self, 
                 min_words=50,
                 max_words=10000,
                 min_avg_word_length=3,
                 max_perplexity=1000):
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_length = min_avg_word_length
        self.max_perplexity = max_perplexity
        
        # Load small LM for perplexity scoring
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.model = AutoModelForCausalLM.from_pretrained('gpt2')
        self.model.eval()
        
    def is_english(self, text):
        """Check if text is in English."""
        try:
            lang = langdetect.detect(text)
            return lang == 'en'
        except:
            return False
    
    def basic_checks(self, text):
        """Basic heuristic checks."""
        words = text.split()
        
        # Word count
        if len(words) < self.min_words or len(words) > self.max_words:
            return False
        
        # Average word length
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < self.min_avg_word_length:
            return False
        
        # Ratio of alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.7:
            return False
        
        return True
    
    def compute_perplexity(self, text):
        """Compute perplexity using GPT-2."""
        encodings = self.tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    
    def filter(self, text):
        """
        Check if text passes quality filters.
        
        Returns:
            (bool, str): (passed, reason)
        """
        # Language check
        if not self.is_english(text):
            return False, "Not English"
        
        # Basic checks
        if not self.basic_checks(text):
            return False, "Failed basic checks"
        
        # Perplexity (computational, optional)
        # ppl = self.compute_perplexity(text)
        # if ppl > self.max_perplexity:
        #     return False, f"High perplexity: {ppl}"
        
        return True, "Passed"

# Example
filter = QualityFilter()
texts = [
    "This is a high-quality English text with proper grammar and sufficient length to be useful for training language models.",
    "bad txt no good",  # Too short, low quality
    "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",  # Gibberish
    "Ceci est un texte en français",  # Not English
]

for text in texts:
    passed, reason = filter.filter(text)
    print(f"{text[:50]}... -> {passed} ({reason})")
```

### 03. Data Augmentation (`03_data_augmentation.md`)

**Techniques for LLMs**:

1. **Back-translation**: Translate to another language and back
2. **Paraphrasing**: Rephrase while keeping meaning
3. **Synonym replacement**: Replace words with synonyms
4. **Sentence reordering**: Shuffle sentence order
5. **Masked token replacement**: BERT-style augmentation

```python
from transformers import MarianMTModel, MarianTokenizer

class DataAugmenter:
    def __init__(self):
        # Back-translation models
        self.en_to_fr_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        self.en_to_fr_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        
        self.fr_to_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
        self.fr_to_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-fr-en')
    
    def back_translate(self, text, intermediate_lang='fr'):
        """Augment via back-translation."""
        # Translate English -> French
        inputs = self.en_to_fr_tokenizer(text, return_tensors='pt', padding=True)
        translated = self.en_to_fr_model.generate(**inputs)
        french_text = self.en_to_fr_tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # Translate French -> English
        inputs = self.fr_to_en_tokenizer(french_text, return_tensors='pt', padding=True)
        back_translated = self.fr_to_en_model.generate(**inputs)
        english_text = self.fr_to_en_tokenizer.decode(back_translated[0], skip_special_tokens=True)
        
        return english_text
    
    def augment_dataset(self, texts, augmentation_factor=2):
        """
        Augment dataset by back-translation.
        
        Args:
            texts: List of original texts
            augmentation_factor: Number of augmented versions per text
        
        Returns:
            Original texts + augmented texts
        """
        augmented = texts.copy()
        
        for text in texts:
            for _ in range(augmentation_factor - 1):
                aug_text = self.back_translate(text)
                augmented.append(aug_text)
        
        return augmented

# Example
augmenter = DataAugmenter()
original = "Machine learning is transforming industries worldwide."
augmented = augmenter.back_translate(original)
print(f"Original: {original}")
print(f"Augmented: {augmented}")
```

### 04. Training Pipelines (`04_training_pipelines.md`)

**Complete Training Loop**:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import wandb

class TextDataset(Dataset):
    """Simple dataset for text data."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class LMTrainer:
    """Language model trainer with all optimizations."""
    
    def __init__(
        self,
        model_name='gpt2',
        learning_rate=5e-5,
        batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=10000,
        warmup_steps=500,
        save_steps=1000,
        logging_steps=100,
        mixed_precision='fp16'
    ):
        # Initialize Accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision
        )
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Training config
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        
        # Initialize wandb
        if self.accelerator.is_main_process:
            wandb.init(project='llm-training', config=self.__dict__)
    
    def create_optimizer(self):
        """Create AdamW optimizer with weight decay."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        return optimizer
    
    def create_scheduler(self, optimizer, num_training_steps):
        """Create learning rate scheduler with warmup."""
        from transformers import get_cosine_schedule_with_warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler
    
    def train(self, train_dataset, eval_dataset=None):
        """Main training loop."""
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Optimizer and scheduler
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer, self.max_steps)
        
        # Prepare with accelerator
        self.model, optimizer, train_dataloader, scheduler = \
            self.accelerator.prepare(
                self.model, optimizer, train_dataloader, scheduler
            )
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0
        
        for epoch in range(100):  # Large number, will break at max_steps
            for batch in train_dataloader:
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Clip gradients
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Logging
                total_loss += loss.item()
                global_step += 1
                
                if global_step % self.logging_steps == 0:
                    avg_loss = total_loss / self.logging_steps
                    
                    if self.accelerator.is_main_process:
                        wandb.log({
                            'loss': avg_loss,
                            'learning_rate': scheduler.get_last_lr()[0],
                            'step': global_step
                        })
                        
                        print(f"Step {global_step}: Loss = {avg_loss:.4f}")
                    
                    total_loss = 0
                
                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")
                
                # Stop at max steps
                if global_step >= self.max_steps:
                    break
            
            if global_step >= self.max_steps:
                break
        
        # Save final model
        self.save_checkpoint("final")
        
        if self.accelerator.is_main_process:
            wandb.finish()
    
    def save_checkpoint(self, checkpoint_name):
        """Save model checkpoint."""
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if self.accelerator.is_main_process:
            unwrapped_model.save_pretrained(checkpoint_name)
            self.tokenizer.save_pretrained(checkpoint_name)
            print(f"Saved checkpoint: {checkpoint_name}")


# Example usage
if __name__ == "__main__":
    # Prepare data
    texts = [
        "This is example text for training.",
        "Language models learn from data.",
        # ... more texts
    ] * 1000  # Simulate larger dataset
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = TextDataset(texts, tokenizer)
    
    # Train
    trainer = LMTrainer(
        model_name='gpt2',
        learning_rate=5e-5,
        batch_size=4,
        gradient_accumulation_steps=8,
        max_steps=1000
    )
    
    trainer.train(dataset)
```

### 05. Fine-Tuning (`05_fine_tuning.md`)

**LoRA (Low-Rank Adaptation)**:

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_lora_model(base_model_name='meta-llama/Llama-2-7b-hf'):
    """Setup model with LoRA adapters."""
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=False,  # Set to True for 8-bit quantization
        device_map='auto'
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank of low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",  # Query projection in attention
            "v_proj",  # Value projection in attention
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    # Output: "trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062"
    
    return model

# Fine-tune only ~0.06% of parameters!
model = setup_lora_model()
```

## ⏱️ Estimated Time

- **Reading**: 5-6 hours
- **Coding**: 8-10 hours
- **Exercises**: 4-5 hours
- **Total**: 17-21 hours

---

**Next**: Module 04 - Reasoning Systems with LLMs →
