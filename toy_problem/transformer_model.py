"""
Small Autoregressive Transformer Model for Toy Problem Case Study

This module implements a simple transformer model to predict tokenized sequences
given conditioning points, as described in the paper's case study.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SimpleTransformer(nn.Module):
    """
    Simple autoregressive transformer for predicting tokenized sequences.
    
    This model takes conditioning points and predicts the next token at each timestep.
    """
    
    def __init__(self, 
                 vocab_size: int = 256,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 max_seq_len: int = 1000,
                 conditioning_dim: int = 8):  # 4 points * 2 (time, value)
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Conditioning embedding (for the 4 conditioning points)
        self.conditioning_embedding = nn.Linear(conditioning_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, 
                token_sequence: torch.Tensor,
                conditioning_points: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            token_sequence: Input token sequence [batch_size, seq_len]
            conditioning_points: Conditioning points [batch_size, 4, 2] (time, value pairs)
            attention_mask: Optional attention mask
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = token_sequence.shape
        
        # Embed tokens
        token_emb = self.token_embedding(token_sequence)  # [batch_size, seq_len, d_model]
        
        # Embed conditioning points
        conditioning_flat = conditioning_points.view(batch_size, -1)  # [batch_size, 8]
        conditioning_emb = self.conditioning_embedding(conditioning_flat)  # [batch_size, d_model]
        
        # Add conditioning to all positions (broadcast)
        conditioning_emb = conditioning_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
        x = token_emb + conditioning_emb
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Create causal mask for autoregressive generation
        if attention_mask is None:
            causal_mask = self.create_causal_mask(seq_len, x.device)
        else:
            causal_mask = attention_mask
        
        # Apply transformer
        x = self.transformer(x, mask=causal_mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate(self, 
                 conditioning_points: torch.Tensor,
                 max_length: int,
                 temperature: float = 1.0,
                 device: torch.device = None) -> torch.Tensor:
        """
        Generate a sequence autoregressively.
        
        Args:
            conditioning_points: Conditioning points [batch_size, 4, 2]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            device: Device to use for generation
            
        Returns:
            generated_tokens: Generated token sequence [batch_size, max_length]
        """
        if device is None:
            device = next(self.parameters()).device
        
        batch_size = conditioning_points.shape[0]
        generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        
        # Start with a special token or zero
        current_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for i in range(max_length):
            # Forward pass
            logits = self.forward(current_tokens, conditioning_points)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token (with numerical stability)
            next_token_logits = torch.clamp(next_token_logits, min=-10, max=10)  # Prevent inf/nan
            probs = F.softmax(next_token_logits, dim=-1)
            # Ensure probabilities are valid
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
            next_token = torch.multinomial(probs, 1)
            
            # Add to generated sequence
            generated[:, i] = next_token.squeeze(-1)
            
            # Update current tokens for next iteration
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        return generated


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleTransformer(
        vocab_size=256,
        d_model=128,
        nhead=8,
        num_layers=4,
        max_seq_len=1000
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 50
    vocab_size = 256
    
    # Create dummy data
    token_sequence = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    conditioning_points = torch.randn(batch_size, 4, 2, device=device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(token_sequence, conditioning_points)
        print(f"Input shape: {token_sequence.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test generation
    with torch.no_grad():
        generated = model.generate(conditioning_points, max_length=100, temperature=1.0, device=device)
        print(f"Generated shape: {generated.shape}")
        print(f"Generated range: [{generated.min()}, {generated.max()}]")
