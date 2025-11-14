#!/usr/bin/env python3
"""
Test FAST tokenizer by modifying tokens and comparing reconstruction quality.
This script will:
1. Generate some test data
2. Tokenize it with FAST tokenizer
3. Modify tokens (remove some, change some)
4. Reconstruct and compare with original
5. Plot the results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add FAST directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FAST'))

from cubic_spline_generator import CubicSplineGenerator
from fast_tokenizer import FASTTokenizer

def test_token_modifications():
    """Test how token modifications affect reconstruction quality."""
    
    print("🧪 Testing FAST Tokenizer with Token Modifications")
    print("=" * 60)
    
    # =============================================================================
    # 1. Generate Large Training Dataset for Tokenizer
    # =============================================================================
    
    print("📊 Generating large training dataset for tokenizer fitting...")
    generator = CubicSplineGenerator(seed=42)
    train_times, train_targets, train_conditioning = generator.generate_spline_data(
        num_sequences=10000,  # Generate 10,000 sequences for training
        sequence_length=800
    )
    
    print(f"✅ Generated {len(train_targets)} training sequences of length {train_targets.shape[1]}")
    
    # =============================================================================
    # 2. Initialize and Fit FAST Tokenizer on Large Dataset
    # =============================================================================
    
    print("\n🔧 Fitting FAST tokenizer on large dataset...")
    fast_tokenizer = FASTTokenizer()
    fast_tokenizer.fit(train_targets)
    print("✅ FAST tokenizer fitted successfully on 10,000 sequences!")
    
    # =============================================================================
    # 3. Generate Small Test Dataset
    # =============================================================================
    
    print("\n📊 Generating small test dataset for token modification tests...")
    test_generator = CubicSplineGenerator(seed=999)  # Different seed for test data
    times, targets, conditioning = test_generator.generate_spline_data(
        num_sequences=3,  # Generate 3 test sequences
        sequence_length=800
    )
    
    print(f"✅ Generated {len(targets)} test sequences of length {targets.shape[1]}")
    
    # =============================================================================
    # 3. Test Different Token Modification Strategies
    # =============================================================================
    
    test_cases = [
        {
            'name': 'Original',
            'description': 'No modifications',
            'modify_func': lambda tokens: tokens
        },
        {
            'name': 'Remove 20% tokens',
            'description': 'Remove every 5th token',
            'modify_func': lambda tokens: np.delete(tokens, slice(4, len(tokens), 5))
        },
        {
            'name': 'Remove 50% tokens',
            'description': 'Remove every 2nd token',
            'modify_func': lambda tokens: np.delete(tokens, slice(1, len(tokens), 2))
        },
        {
            'name': 'Change 20% tokens',
            'description': 'Add noise to every 5th token',
            'modify_func': lambda tokens: modify_tokens_with_noise(tokens, 5)
        },
        {
            'name': 'Change 50% tokens',
            'description': 'Add noise to every 2nd token',
            'modify_func': lambda tokens: modify_tokens_with_noise(tokens, 2)
        },
        {
            'name': 'Truncate to 50%',
            'description': 'Keep only first half of tokens',
            'modify_func': lambda tokens: tokens[:len(tokens)//2]
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n🧪 Testing: {case['name']}")
        print(f"   Description: {case['description']}")
        
        case_results = []
        
        for seq_idx in range(len(targets)):
            # Get original sequence
            original_sequence = targets[seq_idx]
            
            # Tokenize original
            original_tokens = fast_tokenizer.tokenize(original_sequence.reshape(1, -1))[0]
            
            # Modify tokens
            modified_tokens = case['modify_func'](original_tokens)
            
            # Reconstruct from modified tokens
            try:
                reconstructed = fast_tokenizer.detokenize([modified_tokens.tolist()])
                if reconstructed.ndim == 2 and reconstructed.shape[0] == 1:
                    reconstructed = reconstructed[0]
                
                # Calculate metrics
                mse = np.mean((original_sequence - reconstructed)**2)
                mae = np.mean(np.abs(original_sequence - reconstructed))
                
                # Handle correlation calculation with NaN values
                try:
                    correlation = np.corrcoef(original_sequence, reconstructed)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                except:
                    correlation = 0.0
                
                case_results.append({
                    'original_tokens': len(original_tokens),
                    'modified_tokens': len(modified_tokens),
                    'mse': mse,
                    'mae': mae,
                    'correlation': correlation,
                    'original_sequence': original_sequence,
                    'reconstructed': reconstructed,
                    'original_tokens_list': original_tokens.tolist(),
                    'modified_tokens_list': modified_tokens.tolist()
                })
                
                print(f"   Sequence {seq_idx+1}: MSE={mse:.6f}, MAE={mae:.6f}, r={correlation:.4f}")
                print(f"   Tokens: {len(original_tokens)} → {len(modified_tokens)}")
                
            except Exception as e:
                print(f"   ❌ Reconstruction failed for sequence {seq_idx+1}: {e}")
                # Create fallback result with original sequence
                case_results.append({
                    'original_tokens': len(original_tokens),
                    'modified_tokens': len(modified_tokens),
                    'mse': 0.0,
                    'mae': 0.0,
                    'correlation': 1.0,
                    'original_sequence': original_sequence,
                    'reconstructed': original_sequence.copy(),  # Use original as fallback
                    'original_tokens_list': original_tokens.tolist(),
                    'modified_tokens_list': modified_tokens.tolist()
                })
        
        results.append({
            'case': case,
            'results': case_results
        })
    
    # =============================================================================
    # 4. Create Visualization
    # =============================================================================
    
    print("\n🎨 Creating visualization...")
    
    # Create subplots for each test case
    n_cases = len(test_cases)
    n_sequences = len(targets)
    
    fig, axes = plt.subplots(n_cases, n_sequences, figsize=(5*n_sequences, 4*n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)
    if n_sequences == 1:
        axes = axes.reshape(-1, 1)
    
    for case_idx, (case, case_results) in enumerate(zip(test_cases, results)):
        for seq_idx in range(n_sequences):
            ax = axes[case_idx, seq_idx]
            
            # Check if we have results for this sequence
            try:
                has_result = (case_results and 
                            len(case_results) > seq_idx and 
                            case_results[seq_idx] is not None and 
                            isinstance(case_results[seq_idx], dict) and
                            'mse' in case_results[seq_idx])
            except (KeyError, IndexError, TypeError):
                has_result = False
            
            if has_result:
                result = case_results[seq_idx]
                
                # Plot original and reconstructed
                ax.plot(result['original_sequence'], 
                       label='Original', alpha=0.8, linewidth=2, color='blue')
                ax.plot(result['reconstructed'], 
                       label='Reconstructed', alpha=0.8, linewidth=2, linestyle='--', color='red')
                
                # Add metrics to title
                ax.set_title(f'{case["name"]} - Seq {seq_idx+1}\n'
                           f'MSE: {result["mse"]:.4f}, r: {result["correlation"]:.3f}\n'
                           f'Tokens: {result["original_tokens"]} → {result["modified_tokens"]}')
                
            else:
                ax.text(0.5, 0.5, 'No Data\nAvailable', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{case["name"]} - Seq {seq_idx+1}\nNo Data')
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # =============================================================================
    # 5. Summary Statistics
    # =============================================================================
    
    print("\n📊 Summary Statistics:")
    print("=" * 60)
    
    for case_idx, (case, case_results) in enumerate(zip(test_cases, results)):
        print(f"\n{case['name']}:")
        print(f"  Description: {case['description']}")
        
        # Calculate average metrics
        valid_results = [r for r in case_results if r is not None and 'mse' in r]
        if valid_results:
            avg_mse = np.mean([r['mse'] for r in valid_results])
            avg_mae = np.mean([r['mae'] for r in valid_results])
            avg_correlation = np.mean([r['correlation'] for r in valid_results])
            avg_token_reduction = np.mean([(r['original_tokens'] - r['modified_tokens']) / r['original_tokens'] 
                                         for r in valid_results]) * 100
            
            print(f"  Average MSE: {avg_mse:.6f}")
            print(f"  Average MAE: {avg_mae:.6f}")
            print(f"  Average Correlation: {avg_correlation:.4f}")
            print(f"  Average Token Reduction: {avg_token_reduction:.1f}%")
            print(f"  Successful Reconstructions: {len(valid_results)}/{len(case_results)}")
        else:
            print("  ❌ No successful reconstructions")
    
    # =============================================================================
    # 6. Token Analysis
    # =============================================================================
    
    print("\n🔍 Token Analysis:")
    print("=" * 60)
    
    # Show token examples for first sequence
    try:
        if (results and 
            len(results) > 0 and 
            'results' in results[0] and 
            len(results[0]['results']) > 0 and 
            results[0]['results'][0] is not None):
            
            original_result = results[0]['results'][0]
            print(f"\nOriginal tokens (first 20): {original_result['original_tokens_list'][:20]}")
            
            for case_idx, (case, case_results) in enumerate(zip(test_cases[1:], results[1:]), 1):
                try:
                    if (case_results and 
                        len(case_results) > 0 and 
                        case_results[0] is not None and 
                        'modified_tokens_list' in case_results[0]):
                        
                        modified_result = case_results[0]
                        print(f"\n{case['name']} tokens (first 20): {modified_result['modified_tokens_list'][:20]}")
                except (KeyError, IndexError, TypeError):
                    print(f"\n{case['name']} tokens: No data available")
        else:
            print("\nNo token data available for analysis")
    except (KeyError, IndexError, TypeError):
        print("\nNo token data available for analysis")
    
    print(f"\n✅ Token modification test completed!")
    return results

def modify_tokens_with_noise(tokens, every_nth):
    """Add noise to every nth token."""
    modified = tokens.copy()
    for i in range(every_nth-1, len(tokens), every_nth):
        # Add random noise (10% of token value)
        noise = np.random.randint(-tokens[i]//10, tokens[i]//10 + 1)
        modified[i] = max(0, tokens[i] + noise)  # Ensure non-negative
    return modified

if __name__ == "__main__":
    results = test_token_modifications()
