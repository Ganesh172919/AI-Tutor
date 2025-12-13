"""
Complete Transformer Implementation from Scratch
=================================================

This is a full, production-ready transformer implementation in PyTorch,
including encoder, decoder, and encoder-decoder architectures.

Implements all components:
- Multi-head attention
- Position-wise feed-forward networks
- Positional encodings
- Layer normalization
- Residual connections
- Masking (padding and causal)

Based on "Attention Is All You Need" (Vaswani et al., 2017)

Author: LLM Mastery Curriculum
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in the original paper.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This allows the model to learn relative positions.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create position encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encodings added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute output
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional mask tensor
        
        Returns:
            output: Attention output (batch_size, seq_len, d_model)
            attention_weights: Attention weights for visualization
        """
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        # (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for heads
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
        
        # Apply attention
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(x)
        
        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Applied to each position separately and identically.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # FFN(x) = ReLU(xW1 + b1)W2 + b2
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Single layer of the transformer encoder.
    
    Consists of:
    1. Multi-head self-attention
    2. Add & Norm (residual + layer norm)
    3. Position-wise feed-forward
    4. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional padding mask
        
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """
    Single layer of the transformer decoder.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (attending to encoder output)
    4. Add & Norm
    5. Position-wise feed-forward
    6. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            x: Decoder input (batch_size, tgt_len, d_model)
            encoder_output: Encoder output (batch_size, src_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal mask
        
        Returns:
            Output tensor (batch_size, tgt_len, d_model)
        """
        # Self-attention (masked)
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Cross-attention to encoder
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x


class Transformer(nn.Module):
    """
    Complete transformer model (encoder-decoder architecture).
    
    Can be configured as:
    - Encoder-only (BERT-style): set num_decoder_layers=0
    - Decoder-only (GPT-style): set num_encoder_layers=0
    - Encoder-decoder (T5-style): use both
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token IDs (batch_size, src_len)
            src_mask: Source padding mask
        
        Returns:
            Encoder output (batch_size, src_len, d_model)
        """
        # Embed and add positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target token IDs (batch_size, tgt_len)
            encoder_output: Output from encoder
            src_mask: Source padding mask
            tgt_mask: Target causal mask
        
        Returns:
            Decoder output (batch_size, tgt_len, d_model)
        """
        # Embed and add positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through entire transformer.
        
        Args:
            src: Source token IDs (batch_size, src_len)
            tgt: Target token IDs (batch_size, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target causal mask
        
        Returns:
            Output logits (batch_size, tgt_len, tgt_vocab_size)
        """
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(
        self,
        src: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate sequence using greedy or sampling decoding.
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
        
        Returns:
            Generated sequence (batch_size, gen_len)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        encoder_output = self.encode(src)
        
        # Start with BOS token (assuming ID 1)
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Create causal mask
                tgt_len = generated.size(1)
                tgt_mask = self._create_causal_mask(tgt_len).to(device)
                
                # Decode
                decoder_output = self.decode(generated, encoder_output, tgt_mask=tgt_mask)
                
                # Get logits for next token
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences have generated EOS (assuming ID 2)
                if (next_token == 2).all():
                    break
        
        return generated
    
    @staticmethod
    def _create_causal_mask(size: int) -> torch.Tensor:
        """Create causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask == 0
        return mask.unsqueeze(0).unsqueeze(0)


# Example usage and tests
if __name__ == "__main__":
    print("=" * 70)
    print("Transformer Model Test")
    print("=" * 70)
    
    # Model configuration
    config = {
        'src_vocab_size': 10000,
        'tgt_vocab_size': 10000,
        'd_model': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 512,
        'dropout': 0.1
    }
    
    # Create model
    model = Transformer(**config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {num_params:,} parameters")
    
    # Test forward pass
    batch_size = 4
    src_len = 20
    tgt_len = 15
    
    src = torch.randint(0, config['src_vocab_size'], (batch_size, src_len))
    tgt = torch.randint(0, config['tgt_vocab_size'], (batch_size, tgt_len))
    
    print(f"\nInput shapes:")
    print(f"  Source: {src.shape}")
    print(f"  Target: {tgt.shape}")
    
    # Forward pass
    logits = model(src, tgt)
    
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, {tgt_len}, {config['tgt_vocab_size']})")
    
    assert logits.shape == (batch_size, tgt_len, config['tgt_vocab_size']), "Shape mismatch!"
    print("\n✓ Forward pass successful!")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(src[:1], max_length=20, temperature=0.8, top_k=50)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()[:10]}... (first 10)")
    
    print("\n" + "=" * 70)
    print("All tests passed! Transformer is ready to train.")
    print("=" * 70)
