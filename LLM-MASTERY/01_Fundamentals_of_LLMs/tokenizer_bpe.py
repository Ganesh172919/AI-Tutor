"""
Byte Pair Encoding (BPE) Tokenizer Implementation
===================================================

This module implements BPE tokenization from scratch, as used in GPT-2, GPT-3, and RoBERTa.

BPE iteratively merges the most frequent pair of symbols (characters or subwords) in a corpus
to build a vocabulary of subword units optimized for the training data.

Author: LLM Mastery Curriculum
License: MIT
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.
    
    This implementation follows the approach used in GPT-2:
    1. Train on a corpus to learn merge operations
    2. Encode text by applying learned merges
    3. Decode token IDs back to text
    
    Attributes:
        vocab_size (int): Target vocabulary size
        vocab (Dict[str, int]): Mapping from tokens to IDs
        merges (Dict[Tuple[str, str], int]): Learned merge operations
        byte_encoder (Dict[int, str]): Mapping for byte-level BPE
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target size of vocabulary (default: 1000)
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Special tokens
        self.pad_token = '<|endoftext|>'
        self.unk_token = '<|unk|>'
        
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        Create mapping from bytes to Unicode characters.
        
        This is used in GPT-2 style byte-level BPE to ensure all bytes
        can be represented as printable Unicode characters.
        
        Returns:
            Dictionary mapping byte values to Unicode characters
        """
        # Printable ASCII characters
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        
        cs = bs[:]
        n = 0
        
        # Add mappings for non-printable bytes
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of adjacent symbol pairs in vocabulary.
        
        Args:
            vocab: Dictionary mapping word tuples to their frequencies
            
        Returns:
            Dictionary mapping symbol pairs to their total frequency
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        return pairs
    
    def merge_vocab(
        self, 
        pair: Tuple[str, str], 
        vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """
        Merge all occurrences of a symbol pair in the vocabulary.
        
        Args:
            pair: Tuple of two symbols to merge
            vocab: Current vocabulary
            
        Returns:
            Updated vocabulary with merged pairs
        """
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in vocab.items():
            # Convert word tuple to string with spaces
            w = ' '.join(word)
            # Replace the pair
            w = w.replace(bigram, replacement)
            # Convert back to tuple
            new_word = tuple(w.split(' '))
            new_vocab[new_word] = freq
        
        return new_vocab
    
    def train(
        self, 
        corpus: List[str], 
        verbose: bool = False,
        min_frequency: int = 2
    ) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
        """
        Train BPE tokenizer on a corpus.
        
        The training process:
        1. Initialize vocabulary with all characters in the corpus
        2. Count frequencies of adjacent symbol pairs
        3. Iteratively merge the most frequent pair
        4. Repeat until desired vocabulary size is reached
        
        Args:
            corpus: List of text strings to train on
            verbose: Whether to print training progress
            min_frequency: Minimum frequency for a pair to be considered
            
        Returns:
            Tuple of (vocabulary mapping, merge operations)
        """
        if verbose:
            print(f"Training BPE tokenizer on {len(corpus)} documents...")
            print(f"Target vocabulary size: {self.vocab_size}")
        
        # Step 1: Build initial word vocabulary with frequencies
        word_freqs = Counter()
        for text in corpus:
            # Simple whitespace tokenization
            words = re.findall(r'\S+', text.lower())
            word_freqs.update(words)
        
        if verbose:
            print(f"Found {len(word_freqs)} unique words")
        
        # Step 2: Convert words to character sequences with end-of-word marker
        vocab = {}
        for word, freq in word_freqs.items():
            # Represent word as tuple of characters + </w> marker
            vocab[tuple(word) + ('</w>',)] = freq
        
        # Step 3: Get initial character set
        symbols = set()
        for word in vocab.keys():
            symbols.update(word)
        
        # Add special tokens
        symbols.add(self.pad_token)
        symbols.add(self.unk_token)
        
        if verbose:
            print(f"Initial character vocabulary: {len(symbols)} symbols")
        
        # Step 4: Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(symbols)
        
        for i in range(num_merges):
            # Get pair frequencies
            pairs = self.get_stats(vocab)
            
            # Filter by minimum frequency
            pairs = {k: v for k, v in pairs.items() if v >= min_frequency}
            
            if not pairs:
                if verbose:
                    print(f"No more pairs to merge at iteration {i}")
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            if verbose and i % 100 == 0:
                print(f"Merge {i:4d}: {best_pair[0]:10s} + {best_pair[1]:10s} "
                      f"(frequency: {pairs[best_pair]:6d})")
            
            # Store merge operation with its priority (iteration number)
            self.merges[best_pair] = i
            
            # Apply merge to vocabulary
            vocab = self.merge_vocab(best_pair, vocab)
        
        # Step 5: Build final vocabulary mapping
        self.vocab = {}
        idx = 0
        
        # Add special tokens first
        self.vocab[self.pad_token] = idx
        idx += 1
        self.vocab[self.unk_token] = idx
        idx += 1
        
        # Add all symbols from the final vocabulary
        for word in vocab.keys():
            for symbol in word:
                if symbol not in self.vocab:
                    self.vocab[symbol] = idx
                    idx += 1
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Final vocabulary size: {len(self.vocab)}")
            print(f"Number of merge operations: {len(self.merges)}")
        
        return self.vocab, self.merges
    
    def encode(self, text: str, bos: bool = False, eos: bool = True) -> List[int]:
        """
        Encode text into BPE token IDs.
        
        Process:
        1. Split text into words
        2. For each word, start with character-level representation
        3. Apply learned merges in order of priority
        4. Convert final symbols to token IDs
        
        Args:
            text: Input text to tokenize
            bos: Whether to add beginning-of-sequence token
            eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        # Tokenize into words
        words = re.findall(r'\S+', text.lower())
        
        token_ids = []
        
        if bos:
            token_ids.append(self.vocab.get(self.pad_token, 0))
        
        for word in words:
            # Start with character-level representation
            word = tuple(word) + ('</w>',)
            
            # Apply merges
            while len(word) > 1:
                # Get all possible pairs in current word
                pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                
                # Find which pairs have learned merges
                valid_pairs = [(pair, self.merges[pair]) 
                              for pair in pairs if pair in self.merges]
                
                if not valid_pairs:
                    # No more merges possible
                    break
                
                # Apply earliest learned merge (lowest priority number)
                best_pair = min(valid_pairs, key=lambda x: x[1])[0]
                
                # Merge this pair in the word
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        # Merge the pair
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                
                word = tuple(new_word)
            
            # Convert symbols to token IDs
            for symbol in word:
                token_id = self.vocab.get(symbol, self.vocab[self.unk_token])
                token_ids.append(token_id)
        
        if eos:
            token_ids.append(self.vocab.get(self.pad_token, 0))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode BPE token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        # Create reverse vocabulary lookup
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Convert IDs to tokens
        tokens = []
        for tid in token_ids:
            token = id_to_token.get(tid, self.unk_token)
            # Skip special tokens
            if token not in [self.pad_token, self.unk_token]:
                tokens.append(token)
        
        # Join tokens and remove </w> markers
        text = ''.join(tokens).replace('</w>', ' ').strip()
        
        return text
    
    def save(self, filepath: str):
        """
        Save trained tokenizer to file.
        
        Args:
            filepath: Path to save tokenizer
        """
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': [(k, v) for k, v in self.merges.items()],
            'special_tokens': {
                'pad': self.pad_token,
                'unk': self.unk_token
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load trained tokenizer from file.
        
        Args:
            filepath: Path to load tokenizer from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.merges = {tuple(k): v for k, v in data['merges']}
        self.pad_token = data['special_tokens']['pad']
        self.unk_token = data['special_tokens']['unk']
        
        print(f"Tokenizer loaded from {filepath}")
        print(f"Vocabulary size: {len(self.vocab)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into string tokens (not IDs).
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
        """
        token_ids = self.encode(text, bos=False, eos=False)
        id_to_token = {v: k for k, v in self.vocab.items()}
        return [id_to_token.get(tid, self.unk_token) for tid in token_ids]


def demonstrate_bpe():
    """Demonstrate BPE tokenizer with examples."""
    print("=" * 70)
    print("BPE Tokenizer Demonstration")
    print("=" * 70)
    
    # Create training corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox",
        "the lazy dog sleeps",
        "a quick brown fox runs",
        "the dog and the fox",
        "hello world hello hello world",
        "low low low low low",
        "lower lower",
        "lowest lowest lowest",
        "newer newer newer newer",
        "newest newest newest newest newest",
    ] * 10  # Repeat for more training data
    
    print(f"\nTraining on {len(corpus)} sentences...")
    
    # Train tokenizer
    tokenizer = BPETokenizer(vocab_size=200)
    vocab, merges = tokenizer.train(corpus, verbose=True, min_frequency=5)
    
    # Test encoding and decoding
    print("\n" + "=" * 70)
    print("Testing Tokenization")
    print("=" * 70)
    
    test_sentences = [
        "the quick fox",
        "hello world",
        "lowest newer",
        "the lazy brown dog"
    ]
    
    for text in test_sentences:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, bos=False, eos=False)
        decoded = tokenizer.decode(token_ids)
        
        print(f"\nOriginal:  {text}")
        print(f"Tokens:    {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded:   {decoded}")
    
    # Show most common merges
    print("\n" + "=" * 70)
    print("Top 10 Learned Merges")
    print("=" * 70)
    
    sorted_merges = sorted(merges.items(), key=lambda x: x[1])[:10]
    for (token1, token2), priority in sorted_merges:
        print(f"{priority:3d}. '{token1}' + '{token2}' → '{token1}{token2}'")
    
    # Save tokenizer
    print("\n" + "=" * 70)
    tokenizer.save('bpe_tokenizer.json')
    
    # Test loading
    print("\nTesting load...")
    new_tokenizer = BPETokenizer()
    new_tokenizer.load('bpe_tokenizer.json')
    
    test_text = "the quick fox"
    original_tokens = tokenizer.encode(test_text, bos=False, eos=False)
    loaded_tokens = new_tokenizer.encode(test_text, bos=False, eos=False)
    
    assert original_tokens == loaded_tokens, "Loaded tokenizer doesn't match!"
    print("✓ Load test passed!")


def compare_with_huggingface():
    """Compare our BPE with Hugging Face's GPT-2 tokenizer."""
    try:
        from transformers import GPT2Tokenizer
        
        print("\n" + "=" * 70)
        print("Comparing with GPT-2 Tokenizer")
        print("=" * 70)
        
        # Load GPT-2 tokenizer
        hf_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        test_texts = [
            "Hello, world!",
            "This is a test of tokenization.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        for text in test_texts:
            hf_tokens = hf_tokenizer.tokenize(text)
            hf_ids = hf_tokenizer.encode(text)
            
            print(f"\nText: {text}")
            print(f"GPT-2 tokens: {hf_tokens}")
            print(f"GPT-2 IDs: {hf_ids}")
            print(f"Token count: {len(hf_tokens)}")
    
    except ImportError:
        print("\nHugging Face transformers not installed.")
        print("Install with: pip install transformers")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_bpe()
    
    # Compare with HuggingFace if available
    compare_with_huggingface()
    
    print("\n" + "=" * 70)
    print("BPE Tokenizer demonstration complete!")
    print("=" * 70)
