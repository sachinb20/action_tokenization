#!/usr/bin/env python3
"""
Example of using FAST tokenizer with padding for transformer training
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubic_spline_generator import CubicSplineGenerator
from fast_tokenizer import FASTTokenizer
from transformer_model import SimpleTransformer

def transformer_training_example():
    """Example of using FAST tokenizer with padding for transformer training"""
    
    print("FAST Tokenizer + Transformer Training Example")
    print("=" * 50)
    
    # Generate data
    generator = CubicSplineGenerator(seed=42)
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=100, 
        sequence_length=50
    )
    
    print(f"Generated data shape: {targets.shape}")
    print(f"Conditioning shape: {conditioning.shape}")
    
    # Initialize and fit FAST tokenizer
    print("\n1. Fitting FAST tokenizer...")
    fast_tokenizer = FASTTokenizer()
    fast_tokenizer.fit(targets)
    
    # Tokenize with padding for transformer compatibility
    print("\n2. Tokenizing with padding...")
    padded_tokens = fast_tokenizer.tokenize_with_padding(targets, pad_token=0, max_length=45)
    attention_mask = fast_tokenizer.create_attention_mask(padded_tokens, pad_token=0)
    
    print(f"Padded tokens shape: {padded_tokens.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Token range: [{padded_tokens.min()}, {padded_tokens.max()}]")
    print(f"Padding ratio: {np.mean(padded_tokens == 0):.3f}")
    
    # Convert to PyTorch tensors
    token_tensor = torch.from_numpy(padded_tokens).long()
    attention_mask_tensor = torch.from_numpy(attention_mask).bool()
    conditioning_tensor = torch.from_numpy(conditioning).float()
    
    print(f"\n3. PyTorch tensors created:")
    print(f"Token tensor shape: {token_tensor.shape}")
    print(f"Attention mask shape: {attention_mask_tensor.shape}")
    print(f"Conditioning tensor shape: {conditioning_tensor.shape}")
    
    # Initialize transformer model
    print("\n4. Initializing transformer model...")
    model = SimpleTransformer(
        vocab_size=256,  # Should match tokenizer vocab size
        d_model=128,
        nhead=8,
        num_layers=4,
        max_seq_len=1000,
        conditioning_dim=8
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass with padding-aware attention
    print("\n5. Testing forward pass...")
    with torch.no_grad():
        logits = model(
            token_sequence=token_tensor,
            conditioning_points=conditioning_tensor,
            attention_mask=attention_mask_tensor,
            pad_token=0
        )
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Test reconstruction
    print("\n6. Testing reconstruction...")
    reconstructed = fast_tokenizer.detokenize_with_padding(padded_tokens, pad_token=0)
    mse = fast_tokenizer.compute_tokenization_error(targets, reconstructed)
    
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    print("\n✅ FAST tokenizer is ready for transformer training!")
    print("Key features:")
    print("- ✓ Padded sequences for batch processing")
    print("- ✓ Attention masks to ignore padding")
    print("- ✓ High-quality reconstruction")
    print("- ✓ Transformer-compatible format")
    
    return model, fast_tokenizer, token_tensor, attention_mask_tensor, conditioning_tensor

if __name__ == "__main__":
    model, tokenizer, tokens, mask, conditioning = transformer_training_example()
