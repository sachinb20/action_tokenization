#!/usr/bin/env python3
"""
Analyze token sequence lengths for 10,000 sequences of 800 sampling rate
to determine optimal padding length for transformer training.
"""

import numpy as np
import sys
import os
from collections import Counter
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubic_spline_generator import CubicSplineGenerator
from fast_tokenizer import FASTTokenizer

def analyze_token_lengths():
    """Analyze token sequence lengths for large dataset"""
    
    print("🔍 Analyzing Token Sequence Lengths for 10,000 sequences of 800 sampling rate")
    print("=" * 80)
    
    # Parameters
    num_samples = 10000
    sequence_length = 800
    seed = 42
    
    print(f"📊 Dataset Parameters:")
    print(f"  - Number of sequences: {num_samples:,}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Total data points: {num_samples * sequence_length:,}")
    
    # Generate data
    print(f"\n🔄 Generating dataset...")
    generator = CubicSplineGenerator(seed=seed)
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=num_samples,
        sequence_length=sequence_length
    )
    
    print(f"✅ Dataset generated successfully!")
    print(f"  - Shape: {targets.shape}")
    print(f"  - Data range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    # Initialize and fit FAST tokenizer
    print(f"\n🔧 Initializing and fitting FAST tokenizer...")
    fast_tokenizer = FASTTokenizer()
    fast_tokenizer.fit(targets)
    print(f"✅ FAST tokenizer fitted successfully!")
    
    # Tokenize all sequences
    print(f"\n🎯 Tokenizing all {num_samples:,} sequences...")
    tokens = fast_tokenizer.tokenize(targets)
    print(f"✅ Tokenization completed!")
    
    # Analyze token lengths
    print(f"\n📈 Analyzing token sequence lengths...")
    token_lengths = [len(seq) for seq in tokens]
    
    # Basic statistics
    min_length = min(token_lengths)
    max_length = max(token_lengths)
    mean_length = np.mean(token_lengths)
    median_length = np.median(token_lengths)
    std_length = np.std(token_lengths)
    
    print(f"\n📊 Token Length Statistics:")
    print(f"  - Minimum length: {min_length}")
    print(f"  - Maximum length: {max_length}")
    print(f"  - Mean length: {mean_length:.2f}")
    print(f"  - Median length: {median_length:.2f}")
    print(f"  - Standard deviation: {std_length:.2f}")
    
    # Percentile analysis
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    print(f"\n📊 Percentile Analysis:")
    for p in percentiles:
        value = np.percentile(token_lengths, p)
        print(f"  - {p:4.1f}th percentile: {value:.1f}")
    
    # Length distribution
    length_counts = Counter(token_lengths)
    print(f"\n📊 Length Distribution (top 10 most common):")
    for length, count in length_counts.most_common(10):
        percentage = (count / num_samples) * 100
        print(f"  - Length {length:2d}: {count:5d} sequences ({percentage:5.1f}%)")
    
    # Compression analysis
    total_data_points = num_samples * sequence_length
    total_tokens = sum(token_lengths)
    compression_ratio = total_data_points / total_tokens
    
    print(f"\n📊 Compression Analysis:")
    print(f"  - Total data points: {total_data_points:,}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Compression ratio: {compression_ratio:.2f}:1")
    print(f"  - Average tokens per sequence: {total_tokens / num_samples:.2f}")
    
    # Padding length recommendations
    print(f"\n🎯 Padding Length Recommendations:")
    
    # Conservative (99th percentile)
    conservative_length = int(np.percentile(token_lengths, 99))
    print(f"  - Conservative (99th percentile): {conservative_length}")
    
    # Moderate (99.5th percentile)
    moderate_length = int(np.percentile(token_lengths, 99.5))
    print(f"  - Moderate (99.5th percentile): {moderate_length}")
    
    # Aggressive (99.9th percentile)
    aggressive_length = int(np.percentile(token_lengths, 99.9))
    print(f"  - Aggressive (99.9th percentile): {aggressive_length}")
    
    # Buffer factor approach
    buffer_factors = [1.1, 1.2, 1.25, 1.3, 1.5]
    print(f"\n🎯 Buffer Factor Approach:")
    for factor in buffer_factors:
        padded_length = int(max_length * factor)
        sequences_affected = sum(1 for length in token_lengths if length > padded_length)
        print(f"  - Factor {factor:.2f}: {padded_length} tokens (affects {sequences_affected} sequences)")
    
    # Memory usage analysis
    print(f"\n💾 Memory Usage Analysis:")
    for length in [conservative_length, moderate_length, aggressive_length]:
        memory_ratio = length / sequence_length
        padding_overhead = (length - mean_length) / mean_length * 100
        print(f"  - Length {length}: {memory_ratio:.3f}x original, {padding_overhead:.1f}% padding overhead")
    
    # Test different padding lengths
    print(f"\n🧪 Testing Different Padding Lengths:")
    test_lengths = [conservative_length, moderate_length, aggressive_length]
    
    for test_length in test_lengths:
        # Simulate padding
        sequences_that_fit = sum(1 for length in token_lengths if length <= test_length)
        sequences_truncated = num_samples - sequences_that_fit
        truncation_rate = (sequences_truncated / num_samples) * 100
        
        print(f"  - Length {test_length}: {sequences_that_fit:,} sequences fit, {sequences_truncated:,} truncated ({truncation_rate:.2f}%)")
    
    # Final recommendation
    print(f"\n🎯 Final Recommendation:")
    recommended_length = moderate_length  # 99.5th percentile
    print(f"  - Recommended padding length: {recommended_length}")
    print(f"  - Based on: 99.5th percentile")
    print(f"  - Memory efficiency: {recommended_length / sequence_length:.3f}x original")
    print(f"  - Padding overhead: {((recommended_length - mean_length) / mean_length * 100):.1f}%")
    
    # Create visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Histogram of token lengths
        plt.subplot(2, 2, 1)
        plt.hist(token_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.1f}')
        plt.axvline(median_length, color='green', linestyle='--', label=f'Median: {median_length:.1f}')
        plt.axvline(recommended_length, color='orange', linestyle='--', label=f'Recommended: {recommended_length}')
        plt.xlabel('Token Sequence Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Token Sequence Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 2)
        sorted_lengths = np.sort(token_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        plt.plot(sorted_lengths, cumulative, linewidth=2)
        plt.axvline(recommended_length, color='orange', linestyle='--', label=f'Recommended: {recommended_length}')
        plt.xlabel('Token Sequence Length')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Token Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(token_lengths, vert=True)
        plt.ylabel('Token Sequence Length')
        plt.title('Box Plot of Token Lengths')
        plt.grid(True, alpha=0.3)
        
        # Compression ratio over time
        plt.subplot(2, 2, 4)
        compression_ratios = [sequence_length / length for length in token_lengths]
        plt.hist(compression_ratios, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(compression_ratio, color='red', linestyle='--', label=f'Mean: {compression_ratio:.1f}')
        plt.xlabel('Compression Ratio (data points per token)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Compression Ratios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('token_length_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved as 'token_length_analysis.png'")
        
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
    
    return {
        'token_lengths': token_lengths,
        'statistics': {
            'min': min_length,
            'max': max_length,
            'mean': mean_length,
            'median': median_length,
            'std': std_length
        },
        'recommended_length': recommended_length,
        'compression_ratio': compression_ratio
    }

if __name__ == "__main__":
    results = analyze_token_lengths()
    
    print(f"\n✅ Analysis completed!")
    print(f"🎯 Use padding length: {results['recommended_length']}")
    print(f"📊 Compression ratio: {results['compression_ratio']:.2f}:1")
