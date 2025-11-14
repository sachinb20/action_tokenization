"""
FAST Tokenizer-based Transformer Model for Interpolation

This module implements a transformer model optimized for FAST tokenizer-based
interpolation of continuous sequences with proper attention masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # For batch_first=True, keep shape as [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model] for batch_first=True
        # pe shape: [1, max_len, d_model]
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # If sequence is longer than positional encoding, extend it
            self.pe = torch.cat([
                self.pe,
                torch.zeros(1, seq_len - self.pe.size(1), self.pe.size(2), device=x.device)
            ], dim=1)
        return x + self.pe[:, :seq_len, :]


class SimpleTransformer(nn.Module):
    """
    FAST-optimized autoregressive transformer for interpolation.
    
    This model takes conditioning points and predicts the next token at each timestep,
    optimized for FAST tokenizer with proper attention masking and padding handling.
    """
    
    def __init__(self, 
                 vocab_size: int = 2048,  # Default to FAST tokenizer vocab size
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
        # Initialize positional encoding with larger max_len for FAST tokenizer
        self.pos_encoding = PositionalEncoding(d_model, max(max_seq_len, 100))
        
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
                attention_mask: Optional[torch.Tensor] = None,
                pad_token: int = 0) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            token_sequence: Input token sequence [batch_size, seq_len]
            conditioning_points: Conditioning points [batch_size, 4, 2] (time, value pairs)
            attention_mask: Optional attention mask
            pad_token: Token value used for padding (default: 0)
            
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
        
        # Add positional encoding (no transpose needed with batch_first=True)
        x = self.pos_encoding(x)  # [batch_size, seq_len, d_model]
        
        # Create causal mask for autoregressive generation
        if attention_mask is None:
            causal_mask = self.create_causal_mask(seq_len, x.device)
        else:
            # For padding mask, we need to use src_key_padding_mask instead of mask
            # The causal mask should be separate from padding mask
            causal_mask = self.create_causal_mask(seq_len, x.device)
            # Convert attention_mask to src_key_padding_mask format
            src_key_padding_mask = ~attention_mask  # True means "ignore this position"
        
        # Apply transformer
        if attention_mask is None:
            x = self.transformer(x, mask=causal_mask)
        else:
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        return logits
    
    def generate(self, 
                 conditioning_points: torch.Tensor,
                 max_length: int,
                 temperature: float = 1.0,
                 device: torch.device = None,
                 greedy: bool = False,
                 start_tokens: Optional[torch.Tensor] = None,
                 tokenizer: 'FASTTokenizer' = None) -> torch.Tensor:
        """
        Generate a sequence autoregressively.
        
        Args:
            conditioning_points: Conditioning points [batch_size, 4, 2]
            max_length: Maximum length to generate
            temperature: Sampling temperature (ignored if greedy=True)
            device: Device to use for generation
            greedy: If True, use argmax (deterministic). If False, sample from distribution
            start_tokens: Optional starting tokens [batch_size, start_len]. If None, uses conditioning-based start
            tokenizer: Tokenizer to convert conditioning values to tokens. If provided, uses first conditioning point as start token
            
        Returns:
            generated_tokens: Generated token sequence [batch_size, max_length]
        """
        if device is None:
            device = next(self.parameters()).device
        
        batch_size = conditioning_points.shape[0]
        generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        
        # Determine start tokens
        if start_tokens is not None:
            current_tokens = start_tokens.to(device)
        elif tokenizer is not None:
            # Use first conditioning point to determine start token - MUCH BETTER APPROACH!
            first_conditioning_values = conditioning_points[:, 0, 1].cpu().numpy()  # [batch_size] - first conditioning values
            
            # Convert conditioning values to tokens using the actual tokenizer
            # This ensures we use the same tokenization scheme as training!
            # Shape: [batch_size, 1] -> tokenizer expects [batch_size, sequence_length]
            tokenized_sequences = tokenizer.tokenize(first_conditioning_values.reshape(-1, 1))
            # tokenized_sequences is a list of arrays, we need the first token from each
            start_token_indices = np.array([seq[0] if len(seq) > 0 else 0 for seq in tokenized_sequences])
            # Clamp start tokens to valid vocab range
            start_token_indices = np.clip(start_token_indices, 0, self.vocab_size - 1)
            current_tokens = torch.from_numpy(start_token_indices).long().unsqueeze(1).to(device)  # [batch_size, 1]
        else:
            # Fallback: start with empty sequence (less optimal)
            current_tokens = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        
        for i in range(max_length):
            # Forward pass
            if current_tokens.shape[1] == 0:
                # Special case: predict first token from conditioning only (empty sequence fallback)
                dummy_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
                dummy_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                logits = self.forward(
                    token_sequence=dummy_tokens,
                    conditioning_points=conditioning_points,
                    attention_mask=dummy_mask,
                    pad_token=0
                )
                next_token_logits = logits[:, -1, :]
            else:
                # Create attention mask for current tokens (all True since they're real tokens)
                current_mask = torch.ones(batch_size, current_tokens.shape[1], dtype=torch.bool, device=device)
                logits = self.forward(
                    token_sequence=current_tokens,
                    conditioning_points=conditioning_points,
                    attention_mask=current_mask,
                    pad_token=0
                )
                next_token_logits = logits[:, -1, :]
            
            # Select next token
            if greedy or temperature == 0.0:
                # Deterministic: pick most likely token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Stochastic: sample from distribution
                next_token_logits = next_token_logits / temperature
                # Numerical stability
                next_token_logits = torch.clamp(next_token_logits, min=-10, max=10)
                probs = F.softmax(next_token_logits, dim=-1)
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, 1)
            
            # Clamp generated token to valid vocab range
            next_token = torch.clamp(next_token, 0, self.vocab_size - 1)
            
            # Add to generated sequence
            generated[:, i] = next_token.squeeze(-1)
            
            # Update current tokens for next iteration
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        return generated




if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleTransformer(
        vocab_size=2048,  # FAST tokenizer vocabulary size
        d_model=128,
        nhead=8,
        num_layers=4,
        max_seq_len=1000,
        conditioning_dim=8
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 50
    vocab_size = 2048
    
    # Create dummy data
    token_sequence = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    conditioning_points = torch.randn(batch_size, 4, 2, device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(
            token_sequence=token_sequence,
            conditioning_points=conditioning_points,
            attention_mask=attention_mask,
            pad_token=0
        )
        print(f"Input shape: {token_sequence.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test generation
    with torch.no_grad():
        generated = model.generate(conditioning_points, max_length=100, temperature=1.0, device=device)
        print(f"Generated shape: {generated.shape}")
        print(f"Generated range: [{generated.min()}, {generated.max()}]")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }
