#!/usr/bin/env python3
"""
Vanilla FAST Tokenizer Test
===========================
Simple tokenize → decode pipeline without any modifications.
"""

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor
import sys
import os

# Add FAST directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FAST'))
from cubic_spline_generator import CubicSplineGenerator

def generate_test_data(n_sequences=3, sequence_length=800):
    """Generate test cubic spline data."""
    print(f"📊 Generating {n_sequences} test sequences...")
    
    generator = CubicSplineGenerator(seed=42)
    
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

def test_vanilla_reconstruction(tokenizer, sequences):
    """Test vanilla tokenize → decode pipeline."""
    print(f"\n🧪 Testing Vanilla Reconstruction (No Modifications)")
    print("=" * 60)
    
    results = []
    
    for i, sequence in enumerate(sequences):
        print(f"  📊 Sequence {i+1}...")
        
        try:
            # Step 1: Tokenize
            print(f"    🔧 Tokenizing...")
            tokens = tokenizer(sequence.reshape(1, -1, 1))
            print(f"    📊 Tokens: {tokenizer.vocab_size}")
            
            if isinstance(tokens, list):
                tokens_array = np.array(tokens)
                print(f"    📊 Tokens (list): {len(tokens)} items")
            else:
                tokens_array = tokens
                print(f"    📊 Tokens shape: {tokens.shape}")
            
            print(f"    📊 Token range: [{tokens_array.min()}, {tokens_array.max()}]")
            print(f"    📊 Unique tokens: {len(np.unique(tokens_array))}")
            print(f"    🔤 Tokens before modification: {tokens_array.tolist()}")
            
            # Step 2: Decode (reconstruct)
            print(f"    🔧 Decoding...")
            reconstructed = tokenizer.decode(tokens)
            
            # Ensure 1D array
            if reconstructed.ndim > 1:
                reconstructed = reconstructed.flatten()
            
            print(f"    📊 Reconstructed shape: {reconstructed.shape}")
            print(f"    📊 Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
            
            # Step 3: Calculate metrics
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
    fig.suptitle('Vanilla FAST Tokenizer Test\nOriginal vs Reconstructed (No Modifications)', 
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

def main():
    """Main vanilla test function."""
    print("🚀 Vanilla FAST Tokenizer Test")
    print("=" * 50)
    print("📝 This test performs simple tokenize → decode without modifications")
    
    # Load tokenizer
    print("\n📥 Loading pre-trained FAST tokenizer...")
    try:
        tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast",
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Generate test data
    sequences = generate_test_data(n_sequences=3, sequence_length=800)
    
    # Test vanilla reconstruction
    results = test_vanilla_reconstruction(tokenizer, sequences)
    
    # Print summary
    print_summary(results)
    
    # Visualize results
    visualize_results(results)
    
    print(f"\n✅ Vanilla test completed!")

if __name__ == "__main__":
    main()
