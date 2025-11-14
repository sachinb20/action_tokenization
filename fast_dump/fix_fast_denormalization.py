#!/usr/bin/env python3
"""
Fix the FAST tokenizer denormalization issue by adding proper denormalization to the detokenize method.
"""

import numpy as np
import sys
import os

# Add FAST directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FAST'))

from cubic_spline_generator import CubicSplineGenerator
from fast_tokenizer import FASTTokenizer

class FixedFASTTokenizer(FASTTokenizer):
    """FAST tokenizer with fixed denormalization."""
    
    def detokenize(self, tokens):
        """Fixed detokenize method that properly denormalizes data."""
        # Call the original detokenize method
        normalized_data = super().detokenize(tokens)
        
        # Denormalize back to original scale
        try:
            denormalized_data = self._denormalize_data(normalized_data)
            return denormalized_data
        except Exception as e:
            print(f"Warning: Denormalization failed: {e}")
            print(f"Returning normalized data instead")
            return normalized_data
    
    def detokenize_with_padding(self, padded_tokens, pad_token=0):
        """Fixed detokenize_with_padding method that properly denormalizes data."""
        # Call the original detokenize_with_padding method
        normalized_data = super().detokenize_with_padding(padded_tokens, pad_token)
        
        # Denormalize back to original scale
        try:
            denormalized_data = self._denormalize_data(normalized_data)
            return denormalized_data
        except Exception as e:
            print(f"Warning: Denormalization failed: {e}")
            print(f"Returning normalized data instead")
            return normalized_data

def test_fixed_tokenizer():
    """Test the fixed FAST tokenizer."""
    
    print("🔧 Testing Fixed FAST Tokenizer")
    print("=" * 50)
    
    # Generate test data
    print("📊 Generating test data...")
    generator = CubicSplineGenerator(seed=42)
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=3,
        sequence_length=800
    )
    
    print(f"✅ Generated {len(targets)} test sequences")
    print(f"📊 Original data range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    # Test original tokenizer
    print("\n🧪 Testing ORIGINAL FAST tokenizer...")
    original_tokenizer = FASTTokenizer()
    original_tokenizer.fit(targets)
    
    # Test fixed tokenizer
    print("\n🔧 Testing FIXED FAST tokenizer...")
    fixed_tokenizer = FixedFASTTokenizer()
    fixed_tokenizer.fit(targets)
    
    # Compare results
    print("\n📊 Comparing reconstruction quality...")
    
    for i in range(len(targets)):
        print(f"\n--- Sequence {i+1} ---")
        original_sequence = targets[i]
        
        # Test original tokenizer
        try:
            tokens = original_tokenizer.tokenize(original_sequence.reshape(1, -1))[0]
            reconstructed_orig = original_tokenizer.detokenize([tokens.tolist()])
            if reconstructed_orig.ndim == 2 and reconstructed_orig.shape[0] == 1:
                reconstructed_orig = reconstructed_orig[0]
            
            mse_orig = np.mean((original_sequence - reconstructed_orig)**2)
            print(f"  Original tokenizer MSE: {mse_orig:.6f}")
            print(f"  Original reconstructed range: [{reconstructed_orig.min():.3f}, {reconstructed_orig.max():.3f}]")
            
        except Exception as e:
            print(f"  Original tokenizer failed: {e}")
            mse_orig = float('inf')
        
        # Test fixed tokenizer
        try:
            tokens = fixed_tokenizer.tokenize(original_sequence.reshape(1, -1))[0]
            reconstructed_fixed = fixed_tokenizer.detokenize([tokens.tolist()])
            if reconstructed_fixed.ndim == 2 and reconstructed_fixed.shape[0] == 1:
                reconstructed_fixed = reconstructed_fixed[0]
            
            mse_fixed = np.mean((original_sequence - reconstructed_fixed)**2)
            print(f"  Fixed tokenizer MSE: {mse_fixed:.6f}")
            print(f"  Fixed reconstructed range: [{reconstructed_fixed.min():.3f}, {reconstructed_fixed.max():.3f}]")
            
            # Show improvement
            if mse_orig != float('inf'):
                improvement = (mse_orig - mse_fixed) / mse_orig * 100
                print(f"  🎯 Improvement: {improvement:.1f}%")
            
        except Exception as e:
            print(f"  Fixed tokenizer failed: {e}")
    
    print(f"\n✅ Fixed FAST tokenizer test completed!")

if __name__ == "__main__":
    test_fixed_tokenizer()



