#!/usr/bin/env python3
"""
Test FAST tokenizer reconstruction quality to understand the high MSE values.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add FAST directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FAST'))

from cubic_spline_generator import CubicSplineGenerator
from fast_tokenizer import FASTTokenizer

def test_fast_quality():
    """Test FAST tokenizer reconstruction quality."""
    
    print("🔍 Testing FAST Tokenizer Reconstruction Quality")
    print("=" * 60)
    
    # Generate test data
    print("📊 Generating test data...")
    generator = CubicSplineGenerator(seed=42)
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=5,  # Small test set
        sequence_length=800
    )
    
    print(f"✅ Generated {len(targets)} test sequences")
    print(f"📊 Data Statistics:")
    print(f"  - Shape: {targets.shape}")
    print(f"  - Range: [{targets.min():.3f}, {targets.max():.3f}]")
    print(f"  - Mean: {targets.mean():.3f}, Std: {targets.std():.3f}")
    
    # Initialize and fit FAST tokenizer
    print("\n🔧 Fitting FAST tokenizer...")
    fast_tokenizer = FASTTokenizer()
    fast_tokenizer.fit(targets)
    print("✅ FAST tokenizer fitted successfully!")
    
    # Test reconstruction quality for each sequence
    print("\n🧪 Testing reconstruction quality...")
    
    for i in range(len(targets)):
        print(f"\n--- Sequence {i+1} ---")
        
        # Get original sequence
        original = targets[i]
        
        # Tokenize
        tokens = fast_tokenizer.tokenize(original.reshape(1, -1))[0]
        print(f"  Tokens: {len(tokens)} tokens")
        print(f"  Token range: [{min(tokens)}, {max(tokens)}]")
        
        # Detokenize
        try:
            reconstructed = fast_tokenizer.detokenize([tokens.tolist()])
            if reconstructed.ndim == 2 and reconstructed.shape[0] == 1:
                reconstructed = reconstructed[0]
            
            # Calculate metrics
            mse = np.mean((original - reconstructed)**2)
            mae = np.mean(np.abs(original - reconstructed))
            correlation = np.corrcoef(original, reconstructed)[0, 1]
            
            print(f"  ✅ Reconstruction successful")
            print(f"  📊 Metrics:")
            print(f"    - MSE: {mse:.6f}")
            print(f"    - MAE: {mae:.6f}")
            print(f"    - Correlation: {correlation:.4f}")
            print(f"    - Shape: {reconstructed.shape}")
            
            # Check if MSE is reasonable
            if mse > 1.0:
                print(f"  ⚠️  HIGH MSE! This suggests reconstruction issues")
                
                # Analyze the differences
                diff = original - reconstructed
                print(f"  🔍 Difference analysis:")
                print(f"    - Max difference: {np.abs(diff).max():.6f}")
                print(f"    - Mean difference: {np.abs(diff).mean():.6f}")
                print(f"    - Std difference: {np.abs(diff).std():.6f}")
                
                # Check if it's a scaling issue
                print(f"  📏 Scaling analysis:")
                print(f"    - Original range: [{original.min():.3f}, {original.max():.3f}]")
                print(f"    - Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
                
        except Exception as e:
            print(f"  ❌ Reconstruction failed: {e}")
    
    # Test with normalized data
    print(f"\n🔧 Testing with normalized data...")
    try:
        normalized_data = fast_tokenizer._normalize_data(targets)
        print(f"  Normalized data range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
        
        # Tokenize normalized data
        tokens_norm = fast_tokenizer.tokenize(normalized_data)
        reconstructed_norm = fast_tokenizer.detokenize(tokens_norm)
        
        # Compute error on normalized scale
        mse_norm = fast_tokenizer.compute_tokenization_error(normalized_data, reconstructed_norm)
        print(f"  Normalized MSE: {mse_norm:.6f}")
        
        # Compute error on original scale
        mse_orig = fast_tokenizer.compute_tokenization_error(targets, reconstructed_norm)
        print(f"  Original scale MSE: {mse_orig:.6f}")
        
    except Exception as e:
        print(f"  ❌ Normalized test failed: {e}")
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(5, len(targets))):
        ax = axes[i]
        
        # Get original sequence
        original = targets[i]
        
        try:
            # Tokenize and detokenize
            tokens = fast_tokenizer.tokenize(original.reshape(1, -1))[0]
            reconstructed = fast_tokenizer.detokenize([tokens.tolist()])
            if reconstructed.ndim == 2 and reconstructed.shape[0] == 1:
                reconstructed = reconstructed[0]
            
            # Plot
            ax.plot(original, label='Original', alpha=0.8, linewidth=2, color='blue')
            ax.plot(reconstructed, label='Reconstructed', alpha=0.8, linewidth=2, linestyle='--', color='red')
            
            # Calculate MSE for title
            mse = np.mean((original - reconstructed)**2)
            ax.set_title(f'Sequence {i+1}\nMSE: {mse:.4f}')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Sequence {i+1}\nFailed')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    if len(targets) < 5:
        axes[4].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ FAST tokenizer quality test completed!")

if __name__ == "__main__":
    test_fast_quality()



