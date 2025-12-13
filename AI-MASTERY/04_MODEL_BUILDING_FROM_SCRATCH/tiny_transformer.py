"""
Tiny Transformer - Educational Implementation

This is a minimal transformer for learning purposes.
NOT production-ready, but shows all key concepts.

What you'll learn:
- Token embeddings
- Position embeddings  
- Self-attention mechanism
- Feed-forward layers
- Training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TinyAttention(nn.Module):
    """Simplified self-attention"""
    
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq, embed)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Compute attention scores
        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask (can't look at future tokens)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Final projection
        output = self.out_proj(output)
        
        return output


class TinyFeedForward(nn.Module):
    """Simple feed-forward network"""
    
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TinyTransformerBlock(nn.Module):
    """One transformer block: Attention + FFN"""
    
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = TinyAttention(embed_dim, num_heads)
        self.ffn = TinyFeedForward(embed_dim, ff_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Attention with residual connection
        x = x + self.attention(self.ln1(x))
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.ln2(x))
        
        return x


class TinyTransformer(nn.Module):
    """Minimal GPT-style transformer"""
    
    def __init__(self, vocab_size, embed_dim=128, num_layers=4, 
                 num_heads=4, ff_dim=512, max_seq_len=256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_embeds = self.position_embedding(positions)
        
        # Combine token + position
        x = token_embeds + pos_embeds
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits


# Training example
def train_tiny_transformer():
    """Example training loop"""
    
    # Hyperparameters
    vocab_size = 1000  # Small vocabulary
    embed_dim = 128
    num_layers = 4
    batch_size = 8
    seq_len = 64
    num_epochs = 10
    learning_rate = 3e-4
    
    # Create model
    model = TinyTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 100
        
        for batch in range(num_batches):
            # Generate random data (replace with real data)
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Targets are input shifted by one position
            targets = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            # Forward pass
            logits = model(input_ids)
            
            # Reshape for loss calculation
            loss = criterion(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model


# Text generation example
def generate_text(model, start_tokens, max_length=50, temperature=1.0):
    """Generate text autoregressively"""
    model.eval()
    
    current_tokens = start_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits = model(current_tokens)
            
            # Get last token's logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
    
    return current_tokens


# Example usage
if __name__ == "__main__":
    print("Training tiny transformer...")
    print("="*60)
    
    model = train_tiny_transformer()
    
    print("\n" + "="*60)
    print("Generating text...")
    
    # Start with random token
    start_tokens = torch.randint(0, 1000, (1, 10))
    generated = generate_text(model, start_tokens, max_length=20)
    
    print(f"Generated token sequence: {generated[0].tolist()}")
    
    print("\n" + "="*60)
    print("Model architecture:")
    print(model)
    
    print("\n" + "="*60)
    print("Key concepts demonstrated:")
    print("✓ Token embeddings")
    print("✓ Position embeddings")
    print("✓ Self-attention mechanism")
    print("✓ Feed-forward layers")
    print("✓ Residual connections")
    print("✓ Layer normalization")
    print("✓ Causal masking")
    print("✓ Next-token prediction")
    print("✓ Training loop with backpropagation")
    print("✓ Text generation")


# Exercise for the reader:
"""
Try modifying:
1. vocab_size - see how it affects model size
2. embed_dim - affects representation capacity
3. num_layers - deeper = more complex patterns
4. num_heads - multi-head attention benefits
5. temperature - in generation (0.5 = conservative, 2.0 = creative)

Try implementing:
1. Load real text data (not random tokens)
2. Build a simple tokenizer
3. Add dropout for regularization
4. Track validation loss
5. Save/load model checkpoints
"""
