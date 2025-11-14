#!/usr/bin/env python3
"""
Simple FAST Tokenizer Test
==========================
Tests reconstruction quality with token modifications and visualization.
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
    
    # Generate spline data using the correct method
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=n_sequences,
        sequence_length=sequence_length,
        time_range=(0.0, 1.0),
        value_range=(-8.0, 8.0)
    )
    
    print(f"✅ Generated sequences: {targets.shape}")
    print(f"📊 Range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    return targets

def modify_tokens(tokens, modification_type, intensity=0.2):
    """Apply token modifications."""
    tokens_array = np.array(tokens) if isinstance(tokens, list) else tokens
    
    if modification_type == "remove":
        # Remove random tokens
        n_remove = int(len(tokens_array) * intensity)
        remove_indices = np.random.choice(len(tokens_array), n_remove, replace=False)
        modified = np.delete(tokens_array, remove_indices)
        
    elif modification_type == "change":
        # Change random tokens to random values
        n_change = int(len(tokens_array) * intensity)
        change_indices = np.random.choice(len(tokens_array), n_change, replace=False)
        modified = tokens_array.copy()
        modified[change_indices] = np.random.randint(0, 1000, n_change)
        
    else:
        modified = tokens_array
    
    return modified

def test_reconstruction(tokenizer, sequences, modification_type="none", intensity=0.2):
    """Test reconstruction with token modifications."""
    print(f"\n🧪 Testing: {modification_type} (intensity: {intensity})")
    
    results = []
    
    for i, sequence in enumerate(sequences):
        print(f"  📊 Sequence {i+1}...")
        
        try:
            # Tokenize
            tokens = tokenizer(sequence.reshape(1, -1, 1))
            if isinstance(tokens, list):
                tokens_array = np.array(tokens)
            else:
                tokens_array = tokens
            
            # Apply modifications
            if modification_type != "none":
                modified_tokens = modify_tokens(tokens_array, modification_type, intensity)
            else:
                modified_tokens = tokens_array
            
            # Decode
            reconstructed = tokenizer.decode(modified_tokens)
            
            # Ensure 1D array
            if reconstructed.ndim > 1:
                reconstructed = reconstructed.flatten()
            
            # Calculate metrics
            mse = np.mean((sequence - reconstructed) ** 2)
            mae = np.mean(np.abs(sequence - reconstructed))
            correlation = np.corrcoef(sequence, reconstructed)[0, 1]
            
            results.append({
                'original': sequence,
                'reconstructed': reconstructed,
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'tokens_original': len(tokens_array),
                'tokens_modified': len(modified_tokens)
            })
            
            print(f"    ✅ MSE: {mse:.4f}, r: {correlation:.3f}")
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            results.append({
                'original': sequence,
                'reconstructed': None,
                'mse': float('inf'),
                'mae': float('inf'),
                'correlation': 0.0,
                'tokens_original': 0,
                'tokens_modified': 0
            })
    
    return results

def visualize_results(results, modification_type, intensity):
    """Create visualization of results."""
    print(f"\n🎨 Creating visualization...")
    
    n_sequences = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Main title
    fig.suptitle(f'FAST Tokenizer Test: {modification_type.title()} (intensity: {intensity})\nOriginal vs Reconstructed', 
                 fontsize=16, fontweight='bold')
    
    for i, result in enumerate(results[:4]):  # Show first 4
        ax = axes[i]
        
        if result['reconstructed'] is not None:
            # Plot original and reconstructed
            ax.plot(result['original'], 
                   label='Original', alpha=0.8, linewidth=2, color='blue')
            ax.plot(result['reconstructed'], 
                   label='Reconstructed', alpha=0.8, linewidth=2, linestyle='--', color='red')
            
            # Add metrics
            ax.set_title(f'Sequence {i+1}\nMSE: {result["mse"]:.4f}, r: {result["correlation"]:.3f}',
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

def main():
    """Main test function."""
    print("🚀 Simple FAST Tokenizer Test")
    print("=" * 50)
    
    # Load tokenizer
    print("📥 Loading pre-trained FAST tokenizer...")
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
    
    # Test different scenarios
    test_cases = [
        ("none", 0.0),      # No modifications
        ("remove", 0.2),    # Remove 20% of tokens
        ("change", 0.2),    # Change 20% of tokens
        ("remove", 0.5),    # Remove 50% of tokens
    ]
    
    for modification_type, intensity in test_cases:
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {modification_type} (intensity: {intensity})")
        print(f"{'='*60}")
        
        # Run test
        results = test_reconstruction(tokenizer, sequences, modification_type, intensity)
        
        # Calculate summary
        valid_results = [r for r in results if r['reconstructed'] is not None]
        if valid_results:
            avg_mse = np.mean([r['mse'] for r in valid_results])
            avg_correlation = np.mean([r['correlation'] for r in valid_results])
            print(f"\n📊 Summary:")
            print(f"  - Successful reconstructions: {len(valid_results)}/{len(results)}")
            print(f"  - Average MSE: {avg_mse:.6f}")
            print(f"  - Average Correlation: {avg_correlation:.4f}")
        else:
            print(f"\n📊 Summary: No successful reconstructions")
        
        # Visualize
        visualize_results(results, modification_type, intensity)
    
    print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    main()
