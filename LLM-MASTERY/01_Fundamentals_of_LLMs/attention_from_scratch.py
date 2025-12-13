"""
Attention Mechanisms from Scratch
==================================

This module implements various attention mechanisms used in transformers:
1. Scaled Dot-Product Attention (the core of transformers)
2. Multi-Head Attention
3. Self-Attention and Cross-Attention
4. Visualization tools

These implementations use both NumPy (for learning) and PyTorch (for production).

Author: LLM Mastery Curriculum
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# NumPy Implementation (Educational)
# ============================================================================

def scaled_dot_product_attention_numpy(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention in NumPy.
    
    The attention mechanism computes:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Intuition:
    - Q (Query): "What am I looking for?"
    - K (Key): "What do I offer?"
    - V (Value): "What information do I carry?"
    - Q @ K^T: Compute similarity between queries and keys
    - softmax: Convert similarities to probability distribution
    - @ V: Aggregate values weighted by attention
    
    Args:
        Q: Query matrix of shape (batch_size, seq_len, d_k)
        K: Key matrix of shape (batch_size, seq_len, d_k)
        V: Value matrix of shape (batch_size, seq_len, d_v)
        mask: Optional mask of shape (batch_size, seq_len, seq_len)
              True/1 values are masked out (set to -inf before softmax)
    
    Returns:
        output: Attention output of shape (batch_size, seq_len, d_v)
        attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
    """
    # Get dimension of keys (for scaling)
    d_k = Q.shape[-1]
    
    # Step 1: Compute attention scores (Q @ K^T)
    # Shape: (batch_size, seq_len, seq_len)
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    
    # Step 2: Scale by sqrt(d_k) to prevent softmax saturation
    # Without scaling, large d_k leads to very small gradients
    scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply mask (if provided)
    # Used for padding and causal (decoder) masking
    if mask is not None:
        scores = np.where(mask == 0, scores, -1e9)
    
    # Step 4: Apply softmax to get attention weights
    # Each row sums to 1, representing a probability distribution
    attention_weights = softmax_numpy(scores, axis=-1)
    
    # Step 5: Compute weighted sum of values
    # Shape: (batch_size, seq_len, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.
    
    softmax(x) = exp(x) / sum(exp(x))
    
    Subtract max for numerical stability:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def multi_head_attention_numpy(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    num_heads: int = 8,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-Head Attention in NumPy.
    
    Instead of one attention function, we use multiple attention "heads" in parallel.
    Each head learns to attend to different aspects of the input.
    
    Process:
    1. Split Q, K, V into num_heads
    2. Apply attention to each head independently
    3. Concatenate head outputs
    
    Args:
        Q: Query of shape (batch_size, seq_len, d_model)
        K: Key of shape (batch_size, seq_len, d_model)
        V: Value of shape (batch_size, seq_len, d_model)
        num_heads: Number of attention heads
        mask: Optional mask
    
    Returns:
        output: Multi-head attention output
        attention_weights: List of attention weights for each head
    """
    batch_size, seq_len, d_model = Q.shape
    
    # d_k = d_v = d_model / num_heads
    d_k = d_model // num_heads
    
    # Reshape for multi-head attention
    # (batch_size, num_heads, seq_len, d_k)
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Apply attention to each head
    head_outputs = []
    head_weights = []
    
    for i in range(num_heads):
        output, weights = scaled_dot_product_attention_numpy(
            Q[:, i], K[:, i], V[:, i], mask
        )
        head_outputs.append(output)
        head_weights.append(weights)
    
    # Concatenate heads
    # (batch_size, seq_len, d_model)
    output = np.concatenate(head_outputs, axis=-1)
    
    return output, head_weights


# ============================================================================
# PyTorch Implementation (Production)
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention layer in PyTorch.
    
    This is the fundamental building block of transformer attention.
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        Initialize attention layer.
        
        Args:
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            Q: Query tensor (batch_size, seq_len, d_k)
            K: Key tensor (batch_size, seq_len, d_k)
            V: Value tensor (batch_size, seq_len, d_v)
            mask: Optional mask (batch_size, seq_len, seq_len)
        
        Returns:
            output: Attention output (batch_size, seq_len, d_v)
            attention_weights: Attention weights (batch_size, seq_len, seq_len)
        """
        # Get dimension for scaling
        d_k = Q.size(-1)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights (regularization)
        attention_weights = self.dropout(attention_weights)
        
        # Compute output: attention_weights @ V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer as used in transformers.
    
    "Instead of performing a single attention function with d_model-dimensional
    keys, values and queries, we found it beneficial to linearly project the
    queries, keys and values h times with different, learned linear projections."
    - "Attention Is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        # These learn to project the input into different subspaces
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Process:
        1. Project Q, K, V through linear layers
        2. Split into multiple heads
        3. Apply scaled dot-product attention to each head
        4. Concatenate heads
        5. Apply output projection
        
        Args:
            Q: Query (batch_size, seq_len, d_model)
            K: Key (batch_size, seq_len, d_model)
            V: Value (batch_size, seq_len, d_model)
            mask: Optional mask
        
        Returns:
            output: Multi-head attention output (batch_size, seq_len, d_model)
            attention_weights: Attention weights from all heads
        """
        batch_size = Q.size(0)
        
        # Step 1: Linear projections
        # Shape: (batch_size, seq_len, d_model)
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # Step 2: Split into multiple heads
        # Reshape: (batch_size, seq_len, num_heads, d_k)
        # Transpose: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Expand mask for heads if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
        
        # Step 3: Apply attention
        # output: (batch_size, num_heads, seq_len, d_k)
        # attention: (batch_size, num_heads, seq_len, seq_len)
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        # Transpose: (batch_size, seq_len, num_heads, d_k)
        # Reshape: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Step 5: Output projection
        output = self.W_O(output)
        output = self.dropout(output)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-Attention layer (Q, K, V all come from the same source).
    
    Used in encoder and decoder for attending to the input sequence itself.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-attention: Q = K = V = x
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask
        
        Returns:
            output: Self-attention output
            attention_weights: Attention weights
        """
        # Q = K = V = x (self-attention)
        return self.attention(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    Cross-Attention layer (Q from decoder, K and V from encoder).
    
    Used in encoder-decoder architectures where the decoder attends to
    encoder outputs (e.g., in machine translation).
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention: Q from decoder, K and V from encoder.
        
        Args:
            decoder_hidden: Decoder hidden states (batch_size, tgt_len, d_model)
            encoder_output: Encoder outputs (batch_size, src_len, d_model)
            mask: Optional mask
        
        Returns:
            output: Cross-attention output
            attention_weights: Attention weights
        """
        # Q from decoder, K and V from encoder
        return self.attention(decoder_hidden, encoder_output, encoder_output, mask)


# ============================================================================
# Masking Functions
# ============================================================================

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create mask for padded positions.
    
    Masks out padding tokens so they don't contribute to attention.
    
    Args:
        seq: Input sequence (batch_size, seq_len)
        pad_idx: Padding token index (default: 0)
    
    Returns:
        mask: Padding mask (batch_size, 1, 1, seq_len)
    """
    # Create mask: 1 for real tokens, 0 for padding
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(size: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal (look-ahead) mask for decoder self-attention.
    
    Prevents positions from attending to future positions.
    Used in autoregressive generation (GPT-style models).
    
    Args:
        size: Sequence length
        device: Device to create mask on
    
    Returns:
        mask: Causal mask (1, size, size)
    
    Example for size=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    
    Position 0 can only attend to position 0
    Position 1 can attend to positions 0 and 1
    Position 2 can attend to positions 0, 1, and 2
    etc.
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    mask = mask == 0  # Invert: 1 for allowed, 0 for masked
    return mask.unsqueeze(0)


# ============================================================================
# Visualization
# ============================================================================

def visualize_attention(
    attention_weights: torch.Tensor,
    tokens: list,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention matrix (seq_len, seq_len)
        tokens: List of token strings
        save_path: Optional path to save figure
    """
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True,
        square=True,
        linewidths=0.5,
        annot=True,  # Show values
        fmt='.2f'
    )
    
    plt.xlabel('Keys (attending to)')
    plt.ylabel('Queries (attending from)')
    plt.title('Attention Weights Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# Demo and Tests
# ============================================================================

def demo_attention_numpy():
    """Demonstrate attention with NumPy."""
    print("=" * 70)
    print("Scaled Dot-Product Attention Demo (NumPy)")
    print("=" * 70)
    
    # Create sample data
    batch_size = 2
    seq_len = 4
    d_k = 8
    
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    print(f"\nInput shapes:")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    
    # Apply attention
    output, weights = scaled_dot_product_attention_numpy(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    print(f"\nAttention weights (first batch, first query):")
    print(weights[0, 0])
    print(f"Sum: {weights[0, 0].sum():.6f} (should be 1.0)")


def demo_attention_pytorch():
    """Demonstrate attention with PyTorch."""
    print("\n" + "=" * 70)
    print("Multi-Head Attention Demo (PyTorch)")
    print("=" * 70)
    
    # Create sample data
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput shape: {x.shape}")
    
    # Create multi-head attention layer
    attention = MultiHeadAttention(d_model, num_heads)
    
    # Forward pass
    output, weights = attention(x, x, x)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"  (batch_size, num_heads, seq_len, seq_len)")
    
    # Verify attention weights sum to 1
    weights_sum = weights.sum(dim=-1)
    print(f"\nAttention weights sum (should be all 1.0):")
    print(f"  Mean: {weights_sum.mean():.6f}")
    print(f"  Std: {weights_sum.std():.6f}")


def demo_masking():
    """Demonstrate different types of masking."""
    print("\n" + "=" * 70)
    print("Masking Demo")
    print("=" * 70)
    
    # Causal mask
    size = 5
    causal_mask = create_causal_mask(size)
    
    print("\nCausal Mask (size=5):")
    print(causal_mask[0].int())
    print("(1 = can attend, 0 = masked)")
    
    # Padding mask
    seq = torch.tensor([[1, 2, 3, 4, 0, 0], [1, 2, 0, 0, 0, 0]])
    padding_mask = create_padding_mask(seq, pad_idx=0)
    
    print("\nPadding Mask:")
    print("Sequence:", seq)
    print("Mask:", padding_mask.squeeze())


def demo_visualization():
    """Demonstrate attention visualization."""
    print("\n" + "=" * 70)
    print("Attention Visualization Demo")
    print("=" * 70)
    
    # Create sample attention matrix
    tokens = ["The", "cat", "sat", "on", "mat"]
    seq_len = len(tokens)
    
    # Simulate attention pattern (cat pays attention to "sat")
    attention = torch.zeros(seq_len, seq_len)
    attention[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # The -> The
    attention[1] = torch.tensor([0.1, 0.3, 0.5, 0.05, 0.05])  # cat -> sat (strong)
    attention[2] = torch.tensor([0.0, 0.3, 0.5, 0.1, 0.1])  # sat -> sat, cat
    attention[3] = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5])  # on -> on, mat
    attention[4] = torch.tensor([0.1, 0.0, 0.0, 0.3, 0.6])  # mat -> mat
    
    print("\nGenerating attention heatmap...")
    visualize_attention(attention, tokens, 'attention_demo.png')


if __name__ == "__main__":
    # Run all demos
    demo_attention_numpy()
    demo_attention_pytorch()
    demo_masking()
    
    # Uncomment to run visualization demo (requires matplotlib)
    # demo_visualization()
    
    print("\n" + "=" * 70)
    print("Attention mechanisms demonstration complete!")
    print("=" * 70)
