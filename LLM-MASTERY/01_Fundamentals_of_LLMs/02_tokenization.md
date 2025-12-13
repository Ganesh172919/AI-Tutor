# Tokenization: From Text to Numbers

## Why Tokenization Matters

Neural networks can't process text directly—they work with numbers. **Tokenization** is the critical bridge that converts human language into numerical representations that LLMs can process.

But tokenization is more than just "splitting text into words." The choice of tokenization strategy affects:

- **Model performance**: Rare words, compound words, multilingual support
- **Efficiency**: Vocabulary size directly impacts model size and speed
- **Robustness**: Handling typos, new words, code, special characters
- **Compression**: More efficient tokenization = longer context windows

## The Tokenization Challenge

### Naive Approach: Split on Whitespace

```python
text = "Hello, world! How are you?"
tokens = text.split()  # ['Hello,', 'world!', 'How', 'are', 'you?']
```

**Problems**:
1. Punctuation attached to words ('world!' vs 'world')
2. Vocabulary explosion (every typo is a new "word")
3. No handling of rare words or morphology
4. Inefficient for compound words (e.g., "unfortunately" as one token vs. subwords)

### Character-Level Tokenization

```python
text = "Hello"
tokens = list(text)  # ['H', 'e', 'l', 'l', 'o']
```

**Pros**: Small vocabulary (~256 for extended ASCII), no unknown words
**Cons**: Long sequences (10x-100x longer), loses word-level patterns

### Word-Level Tokenization

```python
text = "Hello, world!"
tokens = text.lower().replace(',', '').replace('!', '').split()
# ['hello', 'world']
```

**Pros**: Intuitive, preserves semantic units
**Cons**: Huge vocabulary (100K-1M words), rare word problem, no subword morphology

## Subword Tokenization: The Modern Solution

Modern LLMs use **subword tokenization**—a middle ground between characters and words:

- Common words: Single tokens (`"running"` → `["running"]`)
- Rare words: Multiple subword tokens (`"unfortunately"` → `["un", "fortunate", "ly"]`)
- Unknown words: Fallback to characters or bytes

**Benefits**:
1. **Compact vocabulary**: 30K-100K tokens (vs. 1M+ for words)
2. **Handles rare words**: Decompose into known subwords
3. **Morphological awareness**: Shares representations for "run", "running", "runner"
4. **Multilingual**: Works across languages without language-specific rules

### Three Main Algorithms

1. **Byte Pair Encoding (BPE)**: Used by GPT-2, GPT-3, RoBERTa
2. **WordPiece**: Used by BERT, DistilBERT
3. **Unigram/SentencePiece**: Used by T5, XLNet, Llama

## Byte Pair Encoding (BPE)

### The Intuition

BPE iteratively merges the most frequent pair of symbols in a corpus:

```
Initial:  ['l', 'o', 'w', '</w>']  # "low"
Step 1:   Merge 'l' + 'o' → 'lo'
          ['lo', 'w', '</w>']
Step 2:   Merge 'lo' + 'w' → 'low'
          ['low', '</w>']
```

After many merges, we get a vocabulary of subwords optimized for the training corpus.

### The Algorithm

```python
# Pseudocode
def train_bpe(corpus, num_merges):
    vocab = set(characters_in_corpus)
    
    for i in range(num_merges):
        # Count all adjacent symbol pairs
        pairs = count_pairs(corpus)
        
        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        
        # Merge this pair into a new symbol
        new_symbol = best_pair[0] + best_pair[1]
        vocab.add(new_symbol)
        
        # Update corpus with merged symbol
        corpus = replace_pair(corpus, best_pair, new_symbol)
    
    return vocab
```

### Example Walkthrough

**Corpus**: `["low", "lower", "newest", "widest"]`

**Step 0: Character-level initialization**
```
Word frequencies:
low: 5,  lower: 2,  newest: 6,  widest: 3

Character sequences:
l o w </w>  (5 times)
l o w e r </w>  (2 times)
n e w e s t </w>  (6 times)
w i d e s t </w>  (3 times)
```

**Step 1: Count pairs**
```
Pair frequencies:
('l', 'o'): 7  (5 from "low", 2 from "lower")
('o', 'w'): 7
('w', '</w>'): 5
('w', 'e'): 8  (2 from "lower", 6 from "newest")
('e', 's'): 9
('s', 't'): 9
...
```

**Step 2: Merge most frequent → ('e', 's'): 9**
```
Vocabulary: {a, b, c, ..., z, </w>, 'es'}

Updated corpus:
n e w es t </w>  (6 times)
w i d es t </w>  (3 times)
```

**Step 3: Repeat** for desired vocabulary size (usually 30K-50K merges)

### Full Python Implementation

```python
import re
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        
    def get_pairs(self, word):
        """Get all adjacent pairs of symbols in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def train(self, corpus, verbose=False):
        """
        Train BPE tokenizer on a corpus.
        
        Args:
            corpus: List of strings (sentences or documents)
            verbose: Print progress
        
        Returns:
            vocab: Dictionary mapping tokens to IDs
            merges: Dictionary of merge operations
        """
        # Step 1: Initialize vocabulary with characters
        # Add </w> as word boundary marker
        vocab = Counter()
        for text in corpus:
            words = text.strip().split()
            for word in words:
                # Represent word as tuple of characters + end marker
                word = tuple(word) + ('</w>',)
                vocab[word] += 1
        
        # Initial vocabulary: all characters
        symbols = set()
        for word in vocab.keys():
            symbols.update(word)
        
        # Step 2: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(symbols)
        
        for i in range(num_merges):
            # Count all pairs across all words
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                for pair in self.get_pairs(word):
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            if verbose and i % 100 == 0:
                print(f"Merge {i}: {best_pair} (freq: {pairs[best_pair]})")
            
            # Store merge operation
            self.merges[best_pair] = i
            
            # Apply merge to vocabulary
            vocab = self._merge_vocab(best_pair, vocab)
        
        # Build final vocabulary
        self.vocab = {}
        idx = 0
        for word in vocab.keys():
            for symbol in word:
                if symbol not in self.vocab:
                    self.vocab[symbol] = idx
                    idx += 1
        
        return self.vocab, self.merges
    
    def _merge_vocab(self, pair, vocab):
        """Merge all occurrences of pair in vocabulary."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in vocab.items():
            # Convert word tuple to string
            word_str = ' '.join(word)
            # Replace pair
            word_str = word_str.replace(bigram, replacement)
            # Convert back to tuple
            new_word = tuple(word_str.split())
            new_vocab[new_word] = freq
        
        return new_vocab
    
    def encode(self, text):
        """
        Encode text into BPE tokens.
        
        Args:
            text: String to tokenize
        
        Returns:
            List of token IDs
        """
        # Split into words
        words = text.strip().split()
        tokens = []
        
        for word in words:
            # Start with character-level representation
            word = tuple(word) + ('</w>',)
            
            # Apply learned merges
            while len(word) > 1:
                # Find pairs that exist in merge operations
                pairs = self.get_pairs(word)
                # Get the earliest learned merge
                bigram = min(pairs, 
                           key=lambda pair: self.merges.get(pair, float('inf')))
                
                if bigram not in self.merges:
                    break
                
                # Merge the pair
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == bigram:
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = tuple(new_word)
            
            # Convert symbols to IDs
            for symbol in word:
                if symbol in self.vocab:
                    tokens.append(self.vocab[symbol])
                else:
                    # Unknown token (shouldn't happen with BPE)
                    tokens.append(self.vocab.get('<unk>', 0))
        
        return tokens
    
    def decode(self, token_ids):
        """
        Decode BPE tokens back to text.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded string
        """
        # Reverse vocabulary lookup
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Get tokens
        tokens = [id_to_token.get(tid, '<unk>') for tid in token_ids]
        
        # Join and remove </w> markers
        text = ''.join(tokens).replace('</w>', ' ').strip()
        
        return text


# Example usage
if __name__ == "__main__":
    # Training corpus
    corpus = [
        "low low low low low",
        "lower lower",
        "newest newest newest newest newest newest",
        "widest widest widest",
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox",
        "jumps over the lazy dog"
    ]
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    vocab, merges = tokenizer.train(corpus, verbose=True)
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Test encoding
    test_text = "the newest fox"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nOriginal: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
```

### Real-World BPE: GPT-2 Tokenizer

GPT-2 uses a byte-level BPE variant that:
1. Works directly on bytes (not characters) → Handles any Unicode
2. Vocabulary of 50,257 tokens
3. Pre-processes with regex to avoid merging across category boundaries

```python
# Using Hugging Face's implementation
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Hello, world! This is a test of GPT-2's tokenization."
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")

# Decode back
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Inspect tokens
for token_id in tokens:
    token_str = tokenizer.decode([token_id])
    print(f"{token_id}: '{token_str}'")
```

**Output**:
```
Tokens: [15496, 11, 995, 0, 770, 318, 257, 1332, 286, 402, 11571, 12, 17, 338, 11309, 1634, 13]
Token count: 17

0: 'Hello'
11: ','
995: ' world'
0: '!'
...
```

Notice:
- "Hello" is a single token (common word)
- Spaces are part of tokens (" world" not "world")
- Punctuation is separate

## WordPiece: BERT's Tokenizer

### Key Difference from BPE

Instead of frequency-based merging, WordPiece uses **likelihood-based** merging:

```
Score(pair) = P(pair) / (P(part1) × P(part2))
```

Merge the pair that maximally increases the likelihood of the training corpus.

### Algorithm

```python
def train_wordpiece(corpus, vocab_size):
    # Start with characters
    vocab = set(characters)
    
    while len(vocab) < vocab_size:
        # For each possible merge
        best_score = -inf
        best_pair = None
        
        for pair in all_pairs:
            score = likelihood_score(pair, corpus)
            if score > best_score:
                best_score = score
                best_pair = pair
        
        # Add best pair to vocabulary
        vocab.add(best_pair[0] + best_pair[1])
    
    return vocab
```

### Special Tokens

WordPiece uses `##` prefix to indicate subword tokens:

```
"playing" → ["play", "##ing"]
"unaffected" → ["una", "##ffe", "##cted"]
```

This makes it clear which tokens are word-initial vs. continuations.

### Python Implementation

```python
import re
from collections import defaultdict
import math

class WordPieceTokenizer:
    def __init__(self, vocab_size=1000, unk_token='[UNK]'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = {}
        
    def train(self, corpus, verbose=False):
        """Train WordPiece tokenizer."""
        # Step 1: Count word frequencies
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.strip().split()
            for word in words:
                word_freqs[word] += 1
        
        # Step 2: Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs:
            vocab.update(word)
        
        # Add special tokens
        vocab.add(self.unk_token)
        
        # Step 3: Split words into characters
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word)
        
        # Step 4: Iteratively merge subwords
        while len(vocab) < self.vocab_size:
            # Compute scores for all pairs
            pair_scores = self._compute_pair_scores(splits, word_freqs)
            
            if not pair_scores:
                break
            
            # Get best pair
            best_pair = max(pair_scores, key=pair_scores.get)
            
            if verbose and len(vocab) % 100 == 0:
                print(f"Vocab size: {len(vocab)}, merging {best_pair}")
            
            # Merge this pair
            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)
            
            # Update splits
            splits = self._merge_pair(best_pair, splits)
        
        # Build vocabulary mapping
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        
        return self.vocab
    
    def _compute_pair_scores(self, splits, word_freqs):
        """Compute likelihood scores for all adjacent pairs."""
        # Count occurrences of each symbol and pair
        symbol_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            split = splits[word]
            
            # Count symbols
            for symbol in split:
                symbol_freqs[symbol] += freq
            
            # Count pairs
            for i in range(len(split) - 1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
        
        # Compute scores
        scores = {}
        for pair, freq in pair_freqs.items():
            # Likelihood-based score
            score = freq / (symbol_freqs[pair[0]] * symbol_freqs[pair[1]])
            scores[pair] = score
        
        return scores
    
    def _merge_pair(self, pair, splits):
        """Merge a pair across all words."""
        new_token = pair[0] + pair[1]
        new_splits = {}
        
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and \
                   split[i] == pair[0] and split[i+1] == pair[1]:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        
        return new_splits
    
    def encode(self, text):
        """Encode text using longest-match-first strategy."""
        words = text.strip().split()
        tokens = []
        
        for word in words:
            # Start with full word
            subwords = []
            start = 0
            
            while start < len(word):
                # Find longest matching subword
                end = len(word)
                found = False
                
                while start < end:
                    substr = word[start:end]
                    # Add ## prefix if not at word start
                    if start > 0:
                        substr = '##' + substr
                    
                    if substr in self.vocab:
                        subwords.append(self.vocab[substr])
                        found = True
                        break
                    
                    end -= 1
                
                if not found:
                    # Unknown character
                    subwords.append(self.vocab[self.unk_token])
                    start += 1
                else:
                    start = end
            
            tokens.extend(subwords)
        
        return tokens
    
    def decode(self, token_ids):
        """Decode tokens back to text."""
        # Reverse lookup
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = [id_to_token.get(tid, self.unk_token) for tid in token_ids]
        
        # Join tokens, removing ## markers
        text = ''.join(tokens).replace('##', '')
        
        return text


# Example usage
if __name__ == "__main__":
    corpus = [
        "the quick brown fox",
        "the quick brown dog",
        "quick brown animals",
        "the animals are quick"
    ]
    
    tokenizer = WordPieceTokenizer(vocab_size=200)
    vocab = tokenizer.train(corpus, verbose=True)
    
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Test
    text = "the quick fox"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"\nOriginal: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
```

### Using BERT's Tokenizer

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Unfortunately, tokenization is important!"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
# Output: ['unfortunately', ',', 'token', '##ization', 'is', 'important', '!']

# Notice "tokenization" → ["token", "##ization"]
```

## SentencePiece: Language-Agnostic Tokenization

### Why SentencePiece?

BPE and WordPiece assume **space-separated words**, but:
- Chinese/Japanese don't use spaces
- Arabic joins words
- We want raw text → tokens (no pre-tokenization)

**SentencePiece** treats text as a raw stream and learns boundaries.

### Key Features

1. **Language-agnostic**: No whitespace assumption
2. **Reversible**: `decode(encode(x)) == x` exactly
3. **Subword regularization**: Sample from multiple segmentations (improves robustness)
4. **Two modes**: BPE or Unigram (likelihood-based)

### Unigram Language Model

Instead of greedy merging, Unigram:
1. Starts with a large vocabulary
2. Iteratively removes tokens that least hurt likelihood
3. Final vocabulary: tokens that best compress the corpus

```python
# Simplified Unigram
def train_unigram(corpus, target_vocab_size):
    # Start with all possible substrings as candidates
    vocab = get_all_substrings(corpus)
    
    while len(vocab) > target_vocab_size:
        # Compute likelihood of corpus with current vocab
        loss = compute_loss(corpus, vocab)
        
        # Try removing each token
        losses = {}
        for token in vocab:
            losses[token] = compute_loss(corpus, vocab - {token})
        
        # Remove token with smallest loss increase
        token_to_remove = min(losses, key=losses.get)
        vocab.remove(token_to_remove)
    
    return vocab
```

### Using SentencePiece

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='sp_model',
    vocab_size=8000,
    model_type='bpe',  # or 'unigram'
    character_coverage=0.9995,  # Cover 99.95% of characters
    normalization_rule_name='nmt_nfkc'  # Unicode normalization
)

# Load trained model
sp = spm.SentencePieceProcessor()
sp.load('sp_model.model')

# Encode
text = "This is a test sentence."
tokens = sp.encode_as_pieces(text)
print(f"Tokens: {tokens}")
# Output: ['▁This', '▁is', '▁a', '▁test', '▁sentence', '.']

# Note: ▁ represents space

ids = sp.encode_as_ids(text)
print(f"IDs: {ids}")

# Decode
decoded = sp.decode_pieces(tokens)
print(f"Decoded: {decoded}")
```

### Llama's Tokenizer

Llama uses SentencePiece with BPE:

```python
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

text = "Hello, world!"
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['▁Hello', ',', '▁world', '!']
```

## Handling Special Cases

### Code Tokenization

Code has special requirements:
- Preserve indentation
- Keep operators together (`<=`, `==`)
- Handle strings and comments

```python
# GPT-3's approach: Byte-level BPE
text = "def hello():\n    print('Hi!')"
tokens = tokenizer.encode(text)

# Preserves structure:
# ['def', 'Ġhello', '():', 'Ċ', 'ĠĠĠ', 'Ġprint', "('", 'Hi', "!')"]
# Ċ = newline, Ġ = space
```

### Multilingual Tokenization

```python
# XLM-RoBERTa: Multilingual SentencePiece
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

texts = [
    "Hello, world!",  # English
    "Bonjour le monde!",  # French
    "你好世界",  # Chinese
    "مرحبا بالعالم"  # Arabic
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    print(f"{text}: {tokens}")
```

### Numbers and Dates

Different tokenizers handle numbers differently:

```python
# GPT-2
tokenizer.tokenize("12345")
# ['123', '45'] or ['1', '2', '3', '4', '5']

# Better: Byte-level ensures consistency
```

## Comparison of Tokenization Methods

| Method | Vocabulary Size | Used By | Pros | Cons |
|--------|----------------|---------|------|------|
| **Character** | ~256 | CharRNN | No unknown words | Very long sequences |
| **Word** | 100K-1M | Old NLP | Semantic units | Rare word problem |
| **BPE** | 30K-50K | GPT-2/3, RoBERTa | Balanced, efficient | Greedy merging |
| **WordPiece** | 30K | BERT | Likelihood-based | Requires pre-tokenization |
| **SentencePiece** | 8K-64K | T5, Llama, XLM-R | Language-agnostic | More complex |

## Impact on Model Performance

### Vocabulary Size Trade-offs

**Small vocabulary (8K)**:
- ✅ Fewer embedding parameters
- ✅ Faster training (smaller softmax)
- ❌ Longer sequences (more tokens per word)
- ❌ Less semantic granularity

**Large vocabulary (100K)**:
- ✅ Shorter sequences (fewer tokens per word)
- ✅ Better semantic units
- ❌ More parameters (embedding table)
- ❌ Slower training (large softmax)

**Optimal**: 30K-50K for most use cases

### Rare Words

```python
# Example: "unfortunately" is rare in some corpora

# Word-level: Might be [UNK]
# BPE: ["un", "fortun", "ately"]  or ["unfortun", "ately"]
# Benefit: Model can learn "un-" prefix, "-ately" suffix
```

### Multilingual Models

```python
# XLM-RoBERTa vocabulary: 250K tokens across 100 languages
# Allocates tokens proportional to language data

# English words: ~50K tokens
# Chinese characters: ~20K tokens
# Arabic: ~15K tokens
# ...
```

## Best Practices

### 1. Choose Based on Use Case

- **General text**: BPE or SentencePiece BPE
- **Multilingual**: SentencePiece Unigram
- **Code**: Byte-level BPE (handles special chars)
- **Extreme efficiency**: Unigram with small vocab

### 2. Vocabulary Size

```python
# Rule of thumb
vocab_size = sqrt(corpus_size_in_tokens)

# Typical values
# Small model (100M params): 8K-16K
# Medium model (1B params): 32K
# Large model (100B params): 50K-100K
```

### 3. Handle Unknown Tokens

```python
# Always include special tokens
special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

# BPE: Fallback to byte-level
# WordPiece: [UNK] token
# SentencePiece: Character fallback
```

### 4. Normalization

```python
# Before tokenization
text = text.lower()  # Lowercase
text = text.strip()  # Remove whitespace
text = re.sub(r'\s+', ' ', text)  # Collapse spaces

# Unicode normalization
import unicodedata
text = unicodedata.normalize('NFKC', text)
```

## Advanced Topics

### Byte-Pair Encoding at Byte Level

GPT-2 uses **byte-level BPE**:
- Operates on UTF-8 bytes (not Unicode characters)
- Vocabulary of 256 base tokens (all bytes)
- Can represent any text without [UNK]

```python
# Byte-level encoding
text = "Hello 世界"
bytes_list = list(text.encode('utf-8'))
print(bytes_list)
# [72, 101, 108, 108, 111, 32, 228, 184, 150, 231, 149, 140]

# BPE merges operate on these bytes
```

### Subword Regularization

Instead of deterministic tokenization, sample from distribution:

```python
# SentencePiece with sampling
sp.encode_as_pieces("Hello", enable_sampling=True, alpha=0.1)
# Output 1: ['▁He', 'llo']
# Output 2: ['▁', 'H', 'e', 'll', 'o']
# Output 3: ['▁Hello']

# Benefits:
# - More robust to tokenization errors
# - Data augmentation during training
# - Better generalization
```

### Tokenization-Free Models

Recent research explores **character-level** or **byte-level** models:

**ByT5** (Google, 2021):
- No tokenization, works directly on UTF-8 bytes
- Vocabulary size: 384 (256 bytes + special tokens)
- Trade-off: Longer sequences, more compute

## Debugging Tokenization Issues

### Common Problems

```python
# Problem 1: Inconsistent tokenization
tokenizer.tokenize("hello")  # ['hello']
tokenizer.tokenize("Hello")  # ['H', '##ello']
# Solution: Lowercase during training

# Problem 2: Spaces
tokenizer.tokenize("the cat")  # ['the', 'cat'] or ['the', '▁cat']?
# Solution: Use SentencePiece or byte-level BPE

# Problem 3: Numbers
tokenizer.tokenize("2024")  # ['20', '24'] or ['2', '0', '2', '4']?
# Solution: Pre-process numbers or use byte-level
```

### Visualization

```python
import matplotlib.pyplot as plt
from collections import Counter

# Tokenize corpus
all_tokens = []
for text in corpus:
    all_tokens.extend(tokenizer.tokenize(text))

# Plot token frequency
token_freq = Counter(all_tokens)
top_50 = dict(token_freq.most_common(50))

plt.figure(figsize=(12, 6))
plt.bar(range(len(top_50)), list(top_50.values()))
plt.xticks(range(len(top_50)), list(top_50.keys()), rotation=90)
plt.xlabel('Token')
plt.ylabel('Frequency')
plt.title('Top 50 Tokens by Frequency')
plt.tight_layout()
plt.show()
```

## Summary

**Key Takeaways**:

1. **Tokenization is critical**: Directly affects model performance, efficiency, and robustness
2. **Subword tokenization wins**: BPE, WordPiece, SentencePiece balance vocabulary size and semantic units
3. **No one-size-fits-all**: Choose based on language, domain, and model size
4. **Implementation matters**: Byte-level, character fallback, normalization all impact results

**Modern best practices**:
- Use **byte-level BPE** (GPT-style) for general text and code
- Use **SentencePiece Unigram** for multilingual and low-resource languages
- Vocabulary size: **32K-50K** for most applications
- Always include **special tokens**: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`

**Next steps**: Dive into embeddings to see how these tokens become meaningful vectors → `03_embeddings.md`

---

## Exercises

1. **Implement**: Modify the BPE code to use bytes instead of characters
2. **Experiment**: Train tokenizers on different corpus sizes and compare vocabularies
3. **Analyze**: Compare BPE vs. WordPiece on code vs. natural language
4. **Optimize**: What vocabulary size minimizes model perplexity on your data?

See `exercises.md` for detailed challenges!
