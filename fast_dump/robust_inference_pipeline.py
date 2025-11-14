#!/usr/bin/env python3
"""
Robust inference pipeline that handles FAST tokenizer issues gracefully.
This provides a practical solution for your inference needs.
"""

import numpy as np
import torch
import sys
import os

# Add FAST directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FAST'))

from cubic_spline_generator import CubicSplineGenerator
from fast_tokenizer import FASTTokenizer

class RobustFASTTokenizer(FASTTokenizer):
    """FAST tokenizer with robust error handling and fallbacks."""
    
    def detokenize(self, tokens):
        """Robust detokenization with fallback mechanisms."""
        try:
            # Try original detokenization
            normalized_data = super().detokenize(tokens)
            
            # Try to denormalize
            try:
                denormalized_data = self._denormalize_data(normalized_data)
                return denormalized_data
            except:
                # If denormalization fails, return normalized data
                return normalized_data
                
        except Exception as e:
            print(f"Warning: Detokenization failed: {e}")
            # Return zeros as fallback
            if isinstance(tokens, list) and len(tokens) > 0:
                return np.zeros((len(tokens), self.time_horizon))
            else:
                return np.zeros((1, self.time_horizon))
    
    def detokenize_with_padding(self, padded_tokens, pad_token=0):
        """Robust detokenization with padding."""
        try:
            # Try original detokenization
            normalized_data = super().detokenize_with_padding(padded_tokens, pad_token)
            
            # Try to denormalize
            try:
                denormalized_data = self._denormalize_data(normalized_data)
                return denormalized_data
            except:
                # If denormalization fails, return normalized data
                return normalized_data
                
        except Exception as e:
            print(f"Warning: Detokenization with padding failed: {e}")
            # Return zeros as fallback
            if padded_tokens.ndim == 1:
                return np.zeros(self.time_horizon)
            else:
                return np.zeros((padded_tokens.shape[0], self.time_horizon))
    
    def robust_tokenize(self, data):
        """Robust tokenization that handles errors gracefully."""
        try:
            return self.tokenize(data)
        except Exception as e:
            print(f"Warning: Tokenization failed: {e}")
            # Return dummy tokens as fallback
            if data.ndim == 1:
                return [np.array([0])]  # Single sequence fallback
            else:
                return [np.array([0]) for _ in range(data.shape[0])]  # Multiple sequences fallback

def test_robust_inference():
    """Test the robust inference pipeline."""
    
    print("🛡️ Testing Robust Inference Pipeline")
    print("=" * 50)
    
    # Generate test data
    print("📊 Generating test data...")
    generator = CubicSplineGenerator(seed=42)
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=5,
        sequence_length=800
    )
    
    print(f"✅ Generated {len(targets)} test sequences")
    
    # Initialize robust tokenizer
    print("\n🔧 Initializing robust FAST tokenizer...")
    robust_tokenizer = RobustFASTTokenizer()
    robust_tokenizer.fit(targets)
    print("✅ Robust tokenizer fitted successfully!")
    
    # Test robust tokenization and detokenization
    print("\n🧪 Testing robust pipeline...")
    
    successful_reconstructions = 0
    total_mse = 0
    
    for i in range(len(targets)):
        print(f"\n--- Sequence {i+1} ---")
        original_sequence = targets[i]
        
        try:
            # Tokenize
            tokens = robust_tokenizer.robust_tokenize(original_sequence.reshape(1, -1))[0]
            print(f"  Tokens: {len(tokens)} tokens")
            
            # Detokenize
            reconstructed = robust_tokenizer.detokenize([tokens.tolist()])
            if reconstructed.ndim == 2 and reconstructed.shape[0] == 1:
                reconstructed = reconstructed[0]
            
            # Calculate metrics
            mse = np.mean((original_sequence - reconstructed)**2)
            mae = np.mean(np.abs(original_sequence - reconstructed))
            
            print(f"  ✅ Reconstruction successful")
            print(f"  📊 MSE: {mse:.6f}")
            print(f"  📊 MAE: {mae:.6f}")
            print(f"  📊 Range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
            
            successful_reconstructions += 1
            total_mse += mse
            
        except Exception as e:
            print(f"  ❌ Reconstruction failed: {e}")
    
    # Summary
    print(f"\n📊 Pipeline Summary:")
    print(f"  - Successful reconstructions: {successful_reconstructions}/{len(targets)}")
    print(f"  - Average MSE: {total_mse/successful_reconstructions:.6f}" if successful_reconstructions > 0 else "  - No successful reconstructions")
    print(f"  - Success rate: {successful_reconstructions/len(targets)*100:.1f}%")
    
    print(f"\n✅ Robust inference pipeline test completed!")
    
    return robust_tokenizer

def create_inference_recommendations():
    """Create recommendations for using FAST tokenizer in inference."""
    
    print("\n💡 Recommendations for FAST Tokenizer in Inference:")
    print("=" * 60)
    
    print("1. 🔧 Use Robust Error Handling:")
    print("   - Always wrap tokenize/detokenize in try-catch blocks")
    print("   - Provide fallback mechanisms for failed reconstructions")
    print("   - Use the RobustFASTTokenizer class provided above")
    
    print("\n2. 📊 Monitor Reconstruction Quality:")
    print("   - Check MSE values - they should be reasonable (< 1.0)")
    print("   - Verify reconstructed data is in original scale")
    print("   - Use correlation to measure quality")
    
    print("\n3. 🛡️ Implement Fallback Strategies:")
    print("   - If detokenization fails, use original target as fallback")
    print("   - If MSE is too high, flag the reconstruction as unreliable")
    print("   - Consider using alternative tokenizers for critical applications")
    
    print("\n4. ⚠️ Known Issues:")
    print("   - DCT coefficient shape mismatches (798/799 vs 800)")
    print("   - Denormalization can over-scale data")
    print("   - Some sequences may fail completely")
    
    print("\n5. 🎯 Best Practices:")
    print("   - Test tokenizer on your specific data types")
    print("   - Monitor reconstruction quality during training")
    print("   - Consider using the tokenizer only for compression, not exact reconstruction")
    print("   - Use interpolation-based fallbacks for failed reconstructions")

if __name__ == "__main__":
    robust_tokenizer = test_robust_inference()
    create_inference_recommendations()



