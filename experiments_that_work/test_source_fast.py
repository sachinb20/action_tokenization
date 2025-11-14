#!/usr/bin/env python3
"""
Generate dataset and fit to FAST tokenizer directly
"""

import numpy as np
import sys
import os

# Add parent directory to path to import cubic_spline_generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubic_spline_generator import CubicSplineGenerator
from fast.processing_action_tokenizer import UniversalActionProcessor

def generate_and_fit_dataset():
    """Generate dataset and fit to FAST tokenizer"""
    
    # Parameters
    num_samples = 10000
    sequence_length = 800
    seed = 42
    
    print(f"Generating dataset with {num_samples} samples...")
    
    # Initialize generator
    generator = CubicSplineGenerator(seed=seed)
    
    # Generate data
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=num_samples,
        sequence_length=sequence_length
    )
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {targets.shape}")
    print(f"Data range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    # Prepare data for FAST tokenizer
    # FAST expects action chunks with shape (num_sequences, time_horizon, action_dim)
    # Add action dimension if needed
    if targets.ndim == 2:
        action_data = targets.reshape(targets.shape[0], targets.shape[1], 1)
    else:
        action_data = targets
    
    print(f"Action data shape: {action_data.shape}")
    
    # Fit FAST tokenizer directly
    print("\nFitting FAST tokenizer...")
    fast_processor = UniversalActionProcessor.fit(
        action_data=action_data,
        scale=10,
        vocab_size=1024,
        time_horizon=sequence_length,
        action_dim=1
    )
    
    print("FAST tokenizer fitted successfully!")
    print(f"Scale: {fast_processor.scale}")
    print(f"Vocab size: {fast_processor.vocab_size}")
    print(f"Min token: {fast_processor.min_token}")
    print(f"Time horizon: {fast_processor.time_horizon}")
    print(f"Action dim: {fast_processor.action_dim}")
    
    # Test tokenization on a small sample
    print("\nTesting tokenization on first 5 sequences...")
    test_data = action_data[:5]
    tokens = fast_processor(test_data)
    
    print(f"Tokenized {len(tokens)} sequences")
    print(f"First sequence tokens: {tokens[0][:10]}...")  # Show first 10 tokens
    
    # Test detokenization
    print("\nTesting detokenization...")
    reconstructed = fast_processor.decode(tokens)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Compute reconstruction error
    mse = np.mean((test_data - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    return fast_processor, action_data, tokens, reconstructed

if __name__ == "__main__":
    # Generate dataset and fit FAST tokenizer
    processor, data, tokens, reconstructed = generate_and_fit_dataset()
    
    print(f"\nSummary:")
    print(f"Original data shape: {data.shape}")
    print(f"Number of token sequences: {len(tokens)}")
    print(f"Reconstructed data shape: {reconstructed.shape}")
    print(f"FAST tokenizer ready for use!")
