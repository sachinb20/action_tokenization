#!/usr/bin/env python3
"""
Generate dataset and test FAST tokenization using fast_tokenizer.py
"""

import numpy as np
import sys
import os

# Add parent directory to path to import cubic_spline_generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubic_spline_generator import CubicSplineGenerator
from fast_tokenizer import FASTTokenizer

def generate_and_fit_dataset():
    """Generate dataset and test FAST tokenization using fast_tokenizer.py"""
    
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
    
    # Initialize FAST tokenizer using our wrapper class
    print("\nInitializing FAST tokenizer...")
    fast_tokenizer = FASTTokenizer()
    
    # Fit the tokenizer to our data (handles normalization internally)
    print("\nFitting FAST tokenizer to our data...")
    fast_tokenizer.fit(targets)
    
    print("FAST tokenizer fitted successfully!")
    
    # Test tokenization on a small sample
    print("\nTesting tokenization on first 5 sequences...")
    test_data = targets[:5]
    
    # Test variable-length tokenization
    tokens = fast_tokenizer.tokenize(test_data)
    print(f"Variable-length tokenization: {len(tokens)} sequences")
    
    # Test padded tokenization for transformer compatibility
    print("\nTesting padded tokenization for transformer compatibility...")
    padded_tokens = fast_tokenizer.tokenize_with_padding(test_data, pad_token=0, max_length=45)
    print(f"Padded tokens shape: {padded_tokens.shape}")
    print(f"Padded tokens dtype: {padded_tokens.dtype}")
    print(f"Max length used: 40 tokens")
    
    # Create attention mask
    attention_mask = fast_tokenizer.create_attention_mask(padded_tokens, pad_token=0)
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Attention mask (first sequence): {attention_mask[0]}")
    
    print(f"Tokenized {len(tokens)} sequences")
    
    # Print detailed token information for each trajectory
    print("\nDetailed token analysis:")
    for i, token_seq in enumerate(tokens):
        print(f"\nTrajectory {i+1}:")
        print(f"  - Number of tokens: {len(token_seq)}")
        print(f"  - Token shape: {np.array(token_seq).shape}")
        print(f"  - First 10 tokens: {token_seq[:10]}")
        print(f"  - Last 10 tokens: {token_seq[-10:]}")
        print(f"  - Token range: [{min(token_seq)}, {max(token_seq)}]")
        print(f"  - Unique tokens: {len(set(token_seq))}")
    
    # Overall statistics
    all_tokens = [token for seq in tokens for token in seq]
    print(f"\nOverall token statistics:")
    print(f"  - Total tokens across all sequences: {len(all_tokens)}")
    print(f"  - Average tokens per sequence: {len(all_tokens) / len(tokens):.2f}")
    print(f"  - Token range: [{min(all_tokens)}, {max(all_tokens)}]")
    print(f"  - Unique tokens used: {len(set(all_tokens))}")
    
    # Test detokenization
    print("\nTesting detokenization...")
    print(f"Variable-length tokens type: {type(tokens)}")
    print(f"Number of token sequences: {len(tokens)}")
    print(f"Token sequence lengths: {[len(seq) for seq in tokens]}")
    reconstructed = fast_tokenizer.detokenize(tokens)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test padded detokenization
    print("\nTesting padded detokenization...")
    reconstructed_padded = fast_tokenizer.detokenize_with_padding(padded_tokens, pad_token=0)
    print(f"Padded reconstructed shape: {reconstructed_padded.shape}")
    
    # Compare reconstruction quality
    mse_padded = fast_tokenizer.compute_tokenization_error(test_data, reconstructed_padded)
    print(f"Reconstruction MSE (padded): {mse_padded:.6f}")
    
    # Check if padded and variable-length reconstructions are the same
    mse_comparison = fast_tokenizer.compute_tokenization_error(reconstructed, reconstructed_padded)
    print(f"MSE between variable-length and padded reconstruction: {mse_comparison:.6f}")
    if mse_comparison < 1e-6:
        print("✓ Padded and variable-length reconstructions are identical!")
    else:
        print("⚠️ Padded and variable-length reconstructions differ")
    
    # IMPORTANT: Compare normalized data with reconstructed data
    # The FAST tokenizer works in normalized [-1, 1] space
    print("\nNormalizing test data for fair comparison...")
    normalized_test_data = fast_tokenizer._normalize_data(test_data)
    
    # Compute reconstruction error using normalized data
    mse_normalized = fast_tokenizer.compute_tokenization_error(normalized_test_data, reconstructed)
    print(f"Reconstruction MSE (normalized): {mse_normalized:.6f}")
    
    # Also compute MSE on original scale for reference
    mse_original = fast_tokenizer.compute_tokenization_error(test_data, reconstructed)
    print(f"Reconstruction MSE (original scale): {mse_original:.6f}")
    
    # Analyze marginal information
    print("\nAnalyzing marginal information...")
    analysis = fast_tokenizer.analyze_marginal_information(test_data, sequence_length)
    print(f"Entropy: {analysis['entropy']:.3f}")
    print(f"Compression ratio: {analysis['compression_ratio']:.3f}")
    print(f"Tokens per sequence: {analysis['tokens_per_sequence']:.2f}")
    print(f"Unique tokens: {analysis['unique_tokens']}")
    print(f"Total tokens: {analysis['total_tokens']}")
    
    # Additional debugging information
    print(f"\nDebugging Information:")
    print(f"Original data range: [{test_data.min():.3f}, {test_data.max():.3f}]")
    print(f"Normalized data range: [{normalized_test_data.min():.3f}, {normalized_test_data.max():.3f}]")
    print(f"Reconstructed data range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    print(f"Data variance - Original: {np.var(test_data):.6f}")
    print(f"Data variance - Normalized: {np.var(normalized_test_data):.6f}")
    print(f"Data variance - Reconstructed: {np.var(reconstructed):.6f}")
    
    # Check if reconstructed data is properly normalized
    if np.allclose(reconstructed.min(), -1.0, atol=0.1) and np.allclose(reconstructed.max(), 1.0, atol=0.1):
        print("✓ Reconstructed data is properly normalized to [-1, 1] range")
    else:
        print("⚠️ Reconstructed data is not properly normalized")
    
    # Denormalize reconstructed data for comparison with original
    try:
        denormalized_reconstructed = fast_tokenizer._denormalize_data(reconstructed)
        mse_denormalized = fast_tokenizer.compute_tokenization_error(test_data, denormalized_reconstructed)
        print(f"Reconstruction MSE (denormalized): {mse_denormalized:.6f}")
        print(f"Denormalized reconstructed range: [{denormalized_reconstructed.min():.3f}, {denormalized_reconstructed.max():.3f}]")
        print(f"Denormalized variance: {np.var(denormalized_reconstructed):.6f}")
    except Exception as e:
        print(f"Could not denormalize reconstructed data: {e}")
    
    return fast_tokenizer, targets, tokens, reconstructed

def visualize_reconstruction(original_data, reconstructed_data, num_sequences=4):
    """Visualize original vs reconstructed sequences."""
    import matplotlib.pyplot as plt
    
    print(f"\n🎨 Creating reconstruction visualization...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Main title
    fig.suptitle('FAST Tokenizer Reconstruction Quality\nOriginal vs Reconstructed Signals', 
                 fontsize=16, fontweight='bold')
    
    # Show first 4 sequences
    for i in range(min(num_sequences, len(original_data))):
        ax = axes[i]
        
        # Plot original and reconstructed
        ax.plot(original_data[i], 
               label='Original', alpha=0.8, linewidth=2, color='blue')
        ax.plot(reconstructed_data[i], 
               label='Reconstructed', alpha=0.8, linewidth=2, linestyle='--', color='red')
        
        # Calculate metrics for this sequence
        mse = np.mean((original_data[i] - reconstructed_data[i]) ** 2)
        mae = np.mean(np.abs(original_data[i] - reconstructed_data[i]))
        correlation = np.corrcoef(original_data[i], reconstructed_data[i])[0, 1]
        
        # Add metrics to title
        ax.set_title(f'Sequence {i+1}\nMSE: {mse:.4f}, MAE: {mae:.4f}, r: {correlation:.3f}',
                    fontweight='bold')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Signal Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_sequences, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Reconstruction Summary:")
    print(f"  - Sequences visualized: {min(num_sequences, len(original_data))}")
    print(f"  - Original data shape: {original_data.shape}")
    print(f"  - Reconstructed data shape: {reconstructed_data.shape}")
    
    # Calculate overall metrics
    overall_mse = np.mean((original_data - reconstructed_data) ** 2)
    overall_mae = np.mean(np.abs(original_data - reconstructed_data))
    overall_correlation = np.corrcoef(original_data.flatten(), reconstructed_data.flatten())[0, 1]
    
    print(f"  - Overall MSE: {overall_mse:.6f}")
    print(f"  - Overall MAE: {overall_mae:.6f}")
    print(f"  - Overall Correlation: {overall_correlation:.4f}")

if __name__ == "__main__":
    # Generate dataset and test FAST tokenizer using fast_tokenizer.py
    tokenizer, data, tokens, reconstructed = generate_and_fit_dataset()
    
    print(f"\nSummary:")
    print(f"Original data shape: {data.shape}")
    print(f"Number of token sequences: {len(tokens)}")
    print(f"Reconstructed data shape: {reconstructed.shape}")
    print(f"FAST tokenizer ready for use!")
    print(f"✓ Successfully using fast_tokenizer.py implementation!")
    
    # Add visualization
    visualize_reconstruction(data, reconstructed, num_sequences=4)