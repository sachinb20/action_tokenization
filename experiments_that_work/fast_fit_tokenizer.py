#!/usr/bin/env python3
"""
FAST Tokenizer Fitting and Testing
==================================
Generate 10,000 cubic functions, fit a new FAST tokenizer on them,
and test reconstruction quality using quantile normalization.
"""

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor
import sys
import os

# Add FAST directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FAST'))
from cubic_spline_generator import CubicSplineGenerator

def generate_training_data(n_sequences=10000, sequence_length=800):
    """Generate training cubic spline data for tokenizer fitting."""
    print(f"📊 Generating {n_sequences:,} training sequences...")
    
    generator = CubicSplineGenerator(seed=42)
    
    # Generate spline data
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=n_sequences,
        sequence_length=sequence_length,
        time_range=(0.0, 1.0),
        value_range=(-8.0, 8.0)
    )
    
    print(f"✅ Generated training sequences: {targets.shape}")
    print(f"📊 Range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    return targets

def quantile_normalize(data, target_range=(-1, 1)):
    """Apply quantile normalization to range [target_range[0], target_range[1]]."""
    print(f"🔧 Applying quantile normalization to range {target_range}...")
    
    # Flatten all data for quantile computation
    flat_data = data.flatten()
    
    # Compute quantiles for normalization
    q1 = -8
    q99 = 8
    
    print(f"📊 Quantiles: q1={q1:.3f}, q99={q99:.3f}")
    
    # Normalize to [0, 1] using quantiles
    normalized = np.clip((flat_data - q1) / (q99 - q1), 0, 1)
    
    # Scale to target range
    min_val, max_val = target_range
    normalized = normalized * (max_val - min_val) + min_val
    
    # Reshape back to original shape
    normalized = normalized.reshape(data.shape)
    
    print(f"📊 Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    return normalized, q1, q99

def quantile_denormalize(normalized_data, q1, q99, target_range=(-1, 1)):
    """Reverse quantile normalization."""
    min_val, max_val = target_range
    
    # Scale back to [0, 1]
    data_01 = (normalized_data - min_val) / (max_val - min_val)
    
    # Denormalize using original quantiles
    denormalized = data_01 * (q99 - q1) + q1
    
    return denormalized

def fit_tokenizer(action_dataset, vocab_size=500):
    """Fit a new FAST tokenizer on the action dataset."""
    print(f"\n🔧 Fitting new FAST tokenizer on {len(action_dataset):,} sequences...")
    print(f"📊 Using vocab size: {vocab_size}")
    
    # Load base tokenizer
    print("📥 Loading base FAST tokenizer...")
    base_tokenizer = AutoProcessor.from_pretrained(
        "physical-intelligence/fast",
        trust_remote_code=True
    )
    
    # Fit new tokenizer on the dataset
    print("🔄 Fitting tokenizer on action dataset...")
    new_tokenizer = base_tokenizer.fit(action_dataset, vocab_size=vocab_size)
    
    print("✅ Tokenizer fitting completed!")
    return new_tokenizer

def generate_test_data(n_sequences=3, sequence_length=800):
    """Generate test cubic spline data."""
    print(f"📊 Generating {n_sequences} test sequences...")
    
    generator = CubicSplineGenerator(seed=33384)  # Different seed for test data
    
    # Generate spline data
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=n_sequences,
        sequence_length=sequence_length,
        time_range=(0.0, 1.0),
        value_range=(-8.0, 8.0)
    )
    
    print(f"✅ Generated sequences: {targets.shape}")
    print(f"📊 Range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    return targets

def test_vanilla_reconstruction(tokenizer, sequences, q1, q99):
    """Test vanilla tokenize → decode pipeline with normalization."""
    print(f"\n🧪 Testing Vanilla Reconstruction (With Normalization)")
    print("=" * 60)
    
    results = []
    
    for i, sequence in enumerate(sequences):
        print(f"  📊 Sequence {i+1}...")
        
        try:
            # Step 1: Normalize sequence for tokenization
            print(f"    🔧 Normalizing sequence...")
            normalized_seq = quantile_normalize(sequence.reshape(1, -1), target_range=(-1, 1))[0]
            print(f"    📊 Original range: [{sequence.min():.3f}, {sequence.max():.3f}]")
            print(f"    📊 Normalized range: [{normalized_seq.min():.3f}, {normalized_seq.max():.3f}]")
            
            # Step 2: Tokenize normalized sequence
            print(f"    🔧 Tokenizing...")
            tokens = tokenizer(normalized_seq.reshape(1, -1, 1))
            
            if isinstance(tokens, list):
                tokens_array = np.array(tokens)
                print(f"    📊 Tokens (list): {len(tokens)} items")
            else:
                tokens_array = tokens
                print(f"    📊 Tokens shape: {tokens.shape}")
            
            print(f"    📊 Token range: [{tokens_array.min()}, {tokens_array.max()}]")
            print(f"    📊 Unique tokens: {len(np.unique(tokens_array))}")
            
            # Print tokens before detokenizing
            print(f"    🔤 Tokens before detokenizing: {tokens_array.tolist()}")
            
            # Step 3: Decode (reconstruct normalized)
            print(f"    🔧 Decoding...")
            reconstructed_normalized = tokenizer.decode(tokens)
            # Ensure 1D array
            if reconstructed_normalized.ndim > 1:
                reconstructed_normalized = reconstructed_normalized.flatten()
            
            print(f"    📊 Reconstructed normalized range: [{reconstructed_normalized.min():.3f}, {reconstructed_normalized.max():.3f}]")
            
            # Step 4: Denormalize reconstructed sequence
            print(f"    🔧 Denormalizing...")
            reconstructed = quantile_denormalize(reconstructed_normalized, q1, q99, target_range=(-1, 1))
            print(f"    📊 Reconstructed denormalized range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
            
            # Step 5: Calculate metrics on original scale
            mse = np.mean((sequence - reconstructed) ** 2)
            mae = np.mean(np.abs(sequence - reconstructed))
            correlation = np.corrcoef(sequence, reconstructed)[0, 1]
            
            results.append({
                'original': sequence,
                'reconstructed': reconstructed,
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'tokens': tokens_array,
                'success': True
            })
            
            print(f"    ✅ MSE: {mse:.6f}, MAE: {mae:.6f}, r: {correlation:.4f}")
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            results.append({
                'original': sequence,
                'reconstructed': None,
                'mse': float('inf'),
                'mae': float('inf'),
                'correlation': 0.0,
                'tokens': None,
                'success': False
            })
    
    return results

def visualize_results(results):
    """Create visualization of vanilla reconstruction results."""
    print(f"\n🎨 Creating visualization...")
    
    n_sequences = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Main title
    fig.suptitle('Fitted FAST Tokenizer Test\nOriginal vs Reconstructed (No Modifications)', 
                 fontsize=16, fontweight='bold')
    
    for i, result in enumerate(results[:4]):  # Show first 4
        ax = axes[i]
        
        if result['success'] and result['reconstructed'] is not None:
            # Plot original and reconstructed
            ax.plot(result['original'], 
                   label='Original', alpha=0.8, linewidth=2, color='blue')
            ax.plot(result['reconstructed'], 
                   label='Reconstructed', alpha=0.8, linewidth=2, linestyle='--', color='red')
            
            # Add metrics
            ax.set_title(f'Sequence {i+1}\nMSE: {result["mse"]:.6f}, r: {result["correlation"]:.4f}',
                        fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Reconstruction\nFailed', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Sequence {i+1} - Failed', fontweight='bold')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Signal Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(results), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def print_summary(results):
    """Print summary statistics."""
    print(f"\n📊 Summary Statistics:")
    print("=" * 50)
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print(f"✅ Successful reconstructions: {len(successful_results)}/{len(results)}")
        
        # Calculate averages
        avg_mse = np.mean([r['mse'] for r in successful_results])
        avg_mae = np.mean([r['mae'] for r in successful_results])
        avg_correlation = np.mean([r['correlation'] for r in successful_results])
        
        print(f"📈 Average MSE: {avg_mse:.6f}")
        print(f"📈 Average MAE: {avg_mae:.6f}")
        print(f"📈 Average Correlation: {avg_correlation:.4f}")
        
        # Token statistics
        all_tokens = []
        for r in successful_results:
            if r['tokens'] is not None:
                all_tokens.extend(r['tokens'].flatten())
        
        if all_tokens:
            print(f"🔢 Token Statistics:")
            print(f"  - Total tokens: {len(all_tokens)}")
            print(f"  - Token range: [{min(all_tokens)}, {max(all_tokens)}]")
            print(f"  - Unique tokens: {len(set(all_tokens))}")
            print(f"  - Average tokens per sequence: {len(all_tokens) / len(successful_results):.1f}")
        
        # Quality assessment
        if avg_correlation > 0.99:
            print(f"🎯 Quality: EXCELLENT (correlation > 0.99)")
        elif avg_correlation > 0.95:
            print(f"🎯 Quality: GOOD (correlation > 0.95)")
        elif avg_correlation > 0.90:
            print(f"🎯 Quality: FAIR (correlation > 0.90)")
        else:
            print(f"🎯 Quality: POOR (correlation < 0.90)")
            
    else:
        print(f"❌ No successful reconstructions")

def analyze_token_usage(tokenizer, sequences, q1, q99, expected_vocab_size=500):
    """Analyze unique token usage across a dataset of sequences.

    Normalizes each sequence using (q1, q99), tokenizes, aggregates tokens,
    and reports unique token count and coverage vs expected_vocab_size.
    """
    print("\n🔎 Analyzing token usage on test set...")

    all_tokens = []
    sequence_token_counts = []
    per_sequence_tokens: list[np.ndarray] = []
    for i, sequence in enumerate(sequences):
        # Normalize per training quantiles
        normalized_seq = quantile_normalize(sequence.reshape(1, -1), target_range=(-1, 1))[0]
        tokens = tokenizer(normalized_seq.reshape(1, -1, 1))
        if isinstance(tokens, list):
            tokens_array = np.array(tokens)
        else:
            tokens_array = tokens
        flat = tokens_array.flatten()
        all_tokens.extend(flat.tolist())
        sequence_token_counts.append(len(flat))
        per_sequence_tokens.append(flat)

    unique_tokens = set(all_tokens)
    num_unique = len(unique_tokens)
    coverage = (num_unique / expected_vocab_size) * 100.0 if expected_vocab_size > 0 else float('nan')

    print(f"🔢 Total tokens: {len(all_tokens):,}")
    print(f"🔢 Unique tokens used: {num_unique}/{expected_vocab_size} ({coverage:.2f}%)")
    if num_unique > 0:
        print(f"🔢 Min token: {min(unique_tokens)}, Max token: {max(unique_tokens)}")

    # Plot histogram of number of tokens per sequence
    if len(sequence_token_counts) > 0:
        plt.figure(figsize=(8, 4))
        plt.hist(sequence_token_counts, bins=30, color='steelblue', edgecolor='white', alpha=0.9)
        plt.title('Histogram of token counts per sequence')
        plt.xlabel('Number of tokens')
        plt.ylabel('Number of sequences')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Conditional visualization of detokenized sequences based on token count ranges
    def _plot_examples(indices: list[int], title_prefix: str):
        if not indices:
            return
        n = min(2, len(indices))
        fig, axes = plt.subplots(1, n, figsize=(12, 4))
        if n == 1:
            axes = [axes]
        for ax, idx in zip(axes, indices[:2]):
            # Decode and denormalize
            toks = per_sequence_tokens[idx]
            decoded_norm = tokenizer.decode([toks.tolist()])
            if isinstance(decoded_norm, np.ndarray) and decoded_norm.ndim > 1:
                decoded_norm = decoded_norm.flatten()
            decoded = quantile_denormalize(decoded_norm, q1, q99, target_range=(-1, 1))
            original = sequences[idx]
            # Align lengths
            m = min(len(original), len(decoded))
            original = original[:m]
            decoded = decoded[:m]
            ax.plot(original, label='Original', color='blue', alpha=0.8)
            ax.plot(decoded, label='Detokenized', color='red', alpha=0.8, linestyle='--')
            ax.set_title(f"{title_prefix} | Seq {idx} | tokens={sequence_token_counts[idx]}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.show()

    # Ranges: 40-50, 50-60, >60
    in_40_50 = [i for i, c in enumerate(sequence_token_counts) if 40 <= c < 50]
    in_50_60 = [i for i, c in enumerate(sequence_token_counts) if 50 <= c < 60]
    over_60 = [i for i, c in enumerate(sequence_token_counts) if c >= 60]

    _plot_examples(in_40_50, 'Token count 40-49')
    _plot_examples(in_50_60, 'Token count 50-59')
    _plot_examples(over_60, 'Token count >=60')
    return num_unique, unique_tokens

def main():
    """Main function for fitting and testing FAST tokenizer."""
    print("🚀 FAST Tokenizer Fitting and Testing")
    print("=" * 50)
    print("📝 This script fits a new FAST tokenizer on 10,000 cubic functions")
    print("   and tests reconstruction quality using quantile normalization")
    
    # Step 1: Generate training data
    print(f"\n📊 Step 1: Generating training data...")
    training_data = generate_training_data(n_sequences=10000, sequence_length=800)
    
    # Step 2: Apply quantile normalization
    print(f"\n🔧 Step 2: Applying quantile normalization...")
    normalized_data, q1, q99 = quantile_normalize(training_data, target_range=(-1, 1))
    
    # Step 3: Fit tokenizer
    print(f"\n🔄 Step 3: Fitting tokenizer...")
    fitted_tokenizer = fit_tokenizer(normalized_data, vocab_size=400)
    
    # Step 4: Generate small test data for reconstruction demo
    print(f"\n📊 Step 4: Generating test data (3 sequences) for reconstruction...")
    test_sequences = generate_test_data(n_sequences=3, sequence_length=800)
    
    # Step 5: Test reconstruction with normalization
    print(f"\n🧪 Step 5: Testing reconstruction...")
    results = test_vanilla_reconstruction(fitted_tokenizer, test_sequences, q1, q99)
    
    # Step 6: Print summary
    print_summary(results)
    
    # Step 7: Visualize results
    visualize_results(results)

    # Step 8: Generate large test set and analyze unique token usage
    print(f"\n📊 Step 8: Generating test data (1000 sequences) for token usage analysis...")
    usage_sequences = generate_test_data(n_sequences=10000, sequence_length=800)
    analyze_token_usage(fitted_tokenizer, usage_sequences, q1, q99, expected_vocab_size=400)
    
    print(f"\n✅ FAST tokenizer fitting and testing completed!")

if __name__ == "__main__":
    main()
