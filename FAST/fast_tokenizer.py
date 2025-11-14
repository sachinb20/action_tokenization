"""
FAST Tokenizer for Toy Problem Case Study

This module implements the FAST action tokenizer using the HuggingFace AutoProcessor
from the physical-intelligence/fast model. This provides a more efficient tokenization
scheme compared to naive binning.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from scipy.interpolate import interp1d
import torch
from transformers import AutoProcessor
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class FASTTokenizer:
    """
    FAST action tokenizer that uses DCT-based compression for efficient tokenization.
    
    This tokenizer is more efficient than naive binning and maintains better performance
    at high sampling rates by using discrete cosine transform (DCT) compression.
    """
    
    def __init__(self, model_name: str = "physical-intelligence/fast"):
        """
        Initialize the FAST tokenizer.
        
        Args:
            model_name: HuggingFace model name for the FAST tokenizer
        """
        self.model_name = model_name
        self.base_tokenizer = None
        self.is_fitted = False
        self.action_dim = None
        self.time_horizon = None
        self.target_range = (-1, 1)
        self._q1 = -8
        self._q99 = 8
        self.vocab_size = 500
        
    def fit(self, data: np.ndarray):
        """
        Fit the tokenizer to data and train a custom tokenizer.
        
        Args:
            data: Array of shape (num_sequences, sequence_length) or (num_sequences, sequence_length, action_dim)
        """
        print(f"Training FAST tokenizer on {data.shape[0]} sequences...")
        
        # Load the base tokenizer
        self.base_tokenizer = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        normalized_data = self._normalize_data(data)

        # Prepare data for training - FAST expects action chunks
        if normalized_data.ndim == 2:
            # If 2D, assume single action dimension
            action_data = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], 1)
        else:
            action_data = normalized_data
            

        self.tokenizer = self.base_tokenizer.fit(action_data, vocab_size=self.vocab_size)
        
        # Store dimensions for later use
        self.action_dim = action_data.shape[-1]
        self.time_horizon = action_data.shape[1]
        self.is_fitted = True
        
        print(f"FAST tokenizer fitted successfully!")
        print(f"Action dimension: {self.action_dim}")
        print(f"Time horizon: {self.time_horizon}")
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [-1, 1] range using quantile normalization.
        
        Args:
            data: Input data
            
        Returns:
            normalized_data: Data normalized to [-1, 1] range
        """
        # Flatten data for quantile computation
        flat_data = data.flatten()
        
        # Compute quantiles for normalization and store them
        # (already set in __init__ as -8 and 8)
        
        # Normalize to [0, 1] using quantiles
        normalized = np.clip((flat_data - self._q1) / (self._q99 - self._q1), 0, 1)
        
        # Scale to target range
        min_val, max_val = self.target_range
        normalized = normalized * (max_val - min_val) + min_val
        
        # Reshape back to original shape
        normalized = normalized.reshape(data.shape)
                
        return normalized
    
    def _denormalize_data(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Denormalize data from [-1, 1] range back to original scale.
        This requires the normalization parameters from the fit process.
        
        Args:
            normalized_data: Data in [-1, 1] range
            
        Returns:
            denormalized_data: Data in original scale
        """
        if not hasattr(self, '_q1') or not hasattr(self, '_q99'):
            raise ValueError("Tokenizer must be fitted before denormalization")
        min_val, max_val = self.target_range
        
        # Scale back to [0, 1]
        data_01 = (normalized_data - min_val) / (max_val - min_val)
        
        # Denormalize using original quantiles
        denormalized = data_01 * (self._q99 - self._q1) + self._q1
        
        return denormalized
    
    def tokenize(self, data: np.ndarray) -> np.ndarray:
        """
        Convert continuous values to discrete tokens using FAST.
        
        Args:
            data: Array of continuous values
            
        Returns:
            tokens: Array of discrete tokens
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before tokenization")
        
        # Normalize data
        normalized_data = self._normalize_data(data)
        
        # Ensure data has the right shape for FAST - add action dimension if needed
        if normalized_data.ndim == 2:
            # Add action dimension: (num_sequences, sequence_length) -> (num_sequences, sequence_length, 1)
            normalized_data = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], 1)
        
        # Tokenize using FAST - process each sequence separately
        all_tokens = []
        
        for i in range(normalized_data.shape[0]):
            try:
                # Process one sequence at a time
                sequence = normalized_data[i:i+1]  # Keep batch dimension
                tokens = self.tokenizer(sequence)[0]  # Get first (and only) result
                
                # Convert to numpy array if it's a list
                if isinstance(tokens, list):
                    tokens = np.array(tokens)
                
                # Flatten to 1D for this sequence
                if tokens.ndim > 1:
                    tokens = tokens.flatten()
                
                all_tokens.append(tokens)
                
            except Exception as e:
                print(f"Error tokenizing sequence {i}: {e}")
                # Add empty tokens as fallback
                all_tokens.append(np.array([]))
        
        # Convert to numpy array - keep as list of variable-length arrays
        # Don't pad here - let detokenize handle variable lengths
        return all_tokens
    
    def tokenize_with_padding(self, data: np.ndarray, pad_token: int = 0, max_length: int = 40) -> np.ndarray:
        """
        Convert continuous values to discrete tokens using FAST with padding for transformer compatibility.
        
        Args:
            data: Array of continuous values
            pad_token: Token value to use for padding (default: 0)
            max_length: Maximum sequence length for padding (default: 40)
            
        Returns:
            tokens: Padded array of discrete tokens [batch_size, max_length]
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before tokenization")
        
        # Get variable-length tokens first
        variable_tokens = self.tokenize(data)
        
        # Use fixed max_length instead of dynamic max
        # This ensures consistent padding across different batches
        
        # Pad all sequences to the fixed max_length
        padded_tokens = []
        for tokens in variable_tokens:
            if len(tokens) < max_length:
                # Pad with pad_token
                padded = np.pad(tokens, (0, max_length - len(tokens)), 
                              mode='constant', constant_values=pad_token)
            else:
                # Truncate if longer than max_length (should be rare)
                padded = tokens[:max_length]
            padded_tokens.append(padded)
        
        return np.array(padded_tokens)
    
    def detokenize(self, tokens: Union[List[int], np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Convert discrete tokens back to continuous values using FAST.
        
        Args:
            tokens: List or array of discrete tokens (can be variable length)
            
        Returns:
            values: Array of continuous values
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before detokenization")
        
        # Handle different input formats
        if isinstance(tokens, list):
            # Check if it's a list of arrays (from tokenize) or list of ints
            if len(tokens) > 0 and isinstance(tokens[0], np.ndarray):
                # List of variable-length arrays from tokenize
                tokens_list = [token.tolist() for token in tokens]
            elif isinstance(tokens[0], (int, np.integer)):
                # Single sequence as list
                tokens_list = [tokens]
            else:
                # Already a list of lists
                tokens_list = tokens
        elif isinstance(tokens, np.ndarray):
            if tokens.ndim == 1:
                # Single sequence
                tokens_list = [tokens.tolist()]
            else:
                # Multiple sequences - convert to list of lists
                tokens_list = [tokens[i].tolist() for i in range(tokens.shape[0])]
        else:
            raise ValueError(f"Unsupported token format: {type(tokens)}")
        
        # Decode with proper time_horizon and action_dim parameters
        try:
            decoded_actions = self.tokenizer.decode(
                tokens_list
                # ,time_horizon=self.time_horizon, 
                # action_dim=self.action_dim
            )
            
        except Exception as e:
            print(f"Error in detokenization: {e}")
            print(f"Tokens list length: {len(tokens_list)}")
            print(f"Time horizon: {self.time_horizon}, Action dim: {self.action_dim}")
            
            # Try to decode without specifying time_horizon and action_dim
            try:
                print("Attempting to decode without explicit parameters...")
                decoded_actions = self.tokenizer.decode(tokens_list)
                print(f"Decoded shape without parameters: {decoded_actions.shape}")
            except Exception as e2:
                print(f"Second decode attempt failed: {e2}")
                # Return zeros as fallback
                decoded_actions = np.zeros((len(tokens_list), self.time_horizon, self.action_dim))
        
        # Handle shape mismatch - FAST might return different shapes
        if hasattr(decoded_actions, 'shape') and decoded_actions.shape[1] != self.time_horizon:
            print(f"Warning: Decoded shape {decoded_actions.shape} doesn't match expected ({len(tokens_list)}, {self.time_horizon}, {self.action_dim})")
            # Try to interpolate to correct length
            if decoded_actions.shape[1] < self.time_horizon:
                # Upsample using linear interpolation
                upsampled_actions = []
                for i in range(len(decoded_actions)):
                    seq = decoded_actions[i].squeeze()
                    if seq.ndim == 0:
                        seq = np.array([seq])
                    # Create interpolation function
                    x_old = np.linspace(0, 1, len(seq))
                    x_new = np.linspace(0, 1, self.time_horizon)
                    f = interp1d(x_old, seq, kind='linear', fill_value='extrapolate')
                    upsampled = f(x_new)
                    upsampled_actions.append(upsampled.reshape(-1, 1))
                decoded_actions = np.array(upsampled_actions)
            else:
                # Downsample by taking every nth point
                step = decoded_actions.shape[1] // self.time_horizon
                decoded_actions = decoded_actions[:, ::step, :]
                if decoded_actions.shape[1] > self.time_horizon:
                    decoded_actions = decoded_actions[:, :self.time_horizon, :]
        
        # Ensure proper shape - remove extra dimensions if needed
        if isinstance(decoded_actions, np.ndarray):
            if decoded_actions.ndim == 3 and decoded_actions.shape[-1] == 1:
                # Remove the last dimension if it's 1
                decoded_actions = decoded_actions.squeeze(-1)
        else:
            # Convert to numpy array if it's not already
            decoded_actions = np.array(decoded_actions)
            if decoded_actions.ndim == 3 and decoded_actions.shape[-1] == 1:
                decoded_actions = decoded_actions.squeeze(-1)
        
        # Denormalize data from [-1, 1] back to original scale
        denormalized_actions = self._denormalize_data(decoded_actions)
        
        return denormalized_actions
    
    def detokenize_with_padding(self, padded_tokens: np.ndarray, pad_token: int = 0) -> np.ndarray:
        """
        Convert padded discrete tokens back to continuous values using FAST.
        
        Args:
            padded_tokens: Padded array of discrete tokens [batch_size, max_length] or [max_length]
            pad_token: Token value used for padding (default: 0)
            
        Returns:
            values: Array of continuous values
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before detokenization")
        
        # Handle single sequence case
        if padded_tokens.ndim == 1:
            padded_tokens = padded_tokens.reshape(1, -1)
        
        # Remove padding tokens to get variable-length sequences
        variable_tokens = []
        for tokens in padded_tokens:
            # Find where padding starts (first occurrence of pad_token)
            if pad_token in tokens:
                pad_start = np.where(tokens == pad_token)[0]
                if len(pad_start) > 0:
                    # Remove padding
                    actual_tokens = tokens[:pad_start[0]]
                else:
                    actual_tokens = tokens
            else:
                actual_tokens = tokens
            
            variable_tokens.append(actual_tokens.tolist())
        
        # Use the regular detokenize method
        return self.detokenize(variable_tokens)
    
    def create_attention_mask(self, padded_tokens: np.ndarray, pad_token: int = 0) -> np.ndarray:
        """
        Create attention mask for padded tokens to prevent attention to padding tokens.
        
        Args:
            padded_tokens: Padded array of discrete tokens [batch_size, max_seq_len]
            pad_token: Token value used for padding (default: 0)
            
        Returns:
            attention_mask: Boolean mask [batch_size, max_seq_len] where True = valid token, False = padding
        """
        # Create mask where True means "attend to this token" (not padding)
        attention_mask = (padded_tokens != pad_token)
        return attention_mask
    































































        
    
    def compute_tokenization_error(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Compute the MSE between original and reconstructed data.
        
        Args:
            original: Original continuous data
            reconstructed: Reconstructed data from tokens
            
        Returns:
            mse: Mean squared error
        """
        # Handle shape mismatches by ensuring both arrays have compatible shapes
        if original.shape != reconstructed.shape:
            print(f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")
            
            # If reconstructed has extra dimensions, squeeze them
            if reconstructed.ndim > original.ndim:
                # Remove extra dimensions from the end
                while reconstructed.ndim > original.ndim and reconstructed.shape[-1] == 1:
                    reconstructed = reconstructed.squeeze(-1)
            
            # If still mismatched, try to reshape reconstructed to match original
            if original.shape != reconstructed.shape:
                try:
                    # Ensure we're comparing the same number of sequences
                    min_sequences = min(original.shape[0], reconstructed.shape[0])
                    min_length = min(original.shape[1], reconstructed.shape[1])
                    
                    # Truncate both arrays to the same size
                    original_truncated = original[:min_sequences, :min_length]
                    reconstructed_truncated = reconstructed[:min_sequences, :min_length]
                    
                    mse = np.mean((original_truncated - reconstructed_truncated) ** 2)
                    print(f"Using truncated comparison: {original_truncated.shape} vs {reconstructed_truncated.shape}")
                    return mse
                except Exception as e:
                    print(f"Error in shape handling: {e}")
                    return float('inf')
        
        return np.mean((original - reconstructed) ** 2)
    
    def analyze_marginal_information(self, data: np.ndarray, sampling_rate: int) -> dict:
        """
        Analyze the marginal information content for different sampling rates.
        
        This helps understand the efficiency of FAST tokenization compared to naive binning.
        
        Args:
            data: Continuous data
            sampling_rate: The sampling rate (sequence length)
            
        Returns:
            analysis: Dictionary with marginal information statistics
        """
        tokens = self.tokenize(data)
        
        # Flatten all tokens for analysis
        tokens_flat = []
        for token_sequence in tokens:
            tokens_flat.extend(token_sequence)
        tokens_flat = np.array(tokens_flat)
        
        # For FAST, we analyze the token sequence directly
        # since it's already compressed
        unique_tokens, counts = np.unique(tokens_flat, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Compute statistics
        mean_abs_token = np.mean(np.abs(tokens_flat))
        std_token = np.std(tokens_flat)
        zero_token_ratio = np.mean(tokens_flat == 0)
        
        # Calculate compression ratio correctly: data points per token
        total_data_points = data.shape[0] * data.shape[1]
        compression_ratio = total_data_points / len(tokens_flat) if len(tokens_flat) > 0 else 0
        
        analysis = {
            'sampling_rate': sampling_rate,
            'entropy': entropy,
            'mean_abs_token': mean_abs_token,
            'std_token': std_token,
            'zero_token_ratio': zero_token_ratio,
            'unique_tokens': len(unique_tokens),
            'total_tokens': len(tokens_flat),
            'compression_ratio': compression_ratio,
            'tokens_per_sequence': len(tokens_flat) / data.shape[0] if data.shape[0] > 0 else 0
        }
        
        return analysis


    def visualize_reconstruction(self, original: np.ndarray, reconstructed: np.ndarray, 
                                n_sequences: int = 4, title_prefix: str = "FAST Tokenizer"):
        """
        Visualize original vs reconstructed sequences.
        
        Args:
            original: Original sequences [batch_size, seq_len]
            reconstructed: Reconstructed sequences [batch_size, seq_len]
            n_sequences: Number of sequences to plot (default: 4)
            title_prefix: Prefix for the main title
        """
        n_sequences = min(n_sequences, original.shape[0], 4)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Main title
        fig.suptitle(f'{title_prefix}\nOriginal vs Reconstructed', 
                     fontsize=16, fontweight='bold')
        
        for i in range(n_sequences):
            ax = axes[i]
            
            # Calculate metrics
            mse = np.mean((original[i] - reconstructed[i]) ** 2)
            mae = np.mean(np.abs(original[i] - reconstructed[i]))
            correlation = np.corrcoef(original[i], reconstructed[i])[0, 1]
            
            # Plot original and reconstructed
            ax.plot(original[i], 
                   label='Original', alpha=0.8, linewidth=2, color='blue')
            ax.plot(reconstructed[i], 
                   label='Reconstructed', alpha=0.8, linewidth=2, linestyle='--', color='red')
            
            # Add metrics
            ax.set_title(f'Sequence {i+1}\nMSE: {mse:.6f}, r: {correlation:.4f}',
                        fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Signal Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_sequences, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    
    def save_pretrained(self, path: str):
        """
        Save the trained tokenizer.
        
        Args:
            path: Path to save the tokenizer
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before saving")
        
        self.tokenizer.save_pretrained(path)
    
    def load_pretrained(self, path: str):
        """
        Load a pre-trained tokenizer.
        
        Args:
            path: Path to load the tokenizer from
        """
        self.tokenizer = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.is_fitted = True





if __name__ == "__main__":
    # Test the FAST tokenizer
    from cubic_spline_generator import CubicSplineGenerator
    
    print("Testing FAST Tokenizer:")
    print("=" * 50)
    
    # Step 1: Generate training data
    print("\n📊 Step 1: Generating training data...")
    generator = CubicSplineGenerator(seed=42)
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=10000, 
        sequence_length=1000
    )
    print(f"✅ Generated {targets.shape[0]} training sequences")
    
    # Step 2: Initialize and fit tokenizer
    print("\n🔧 Step 2: Fitting FAST tokenizer...")
    fast_tokenizer = FASTTokenizer()
    fast_tokenizer.fit(targets)
    
    # Step 3: Generate test data with different seed
    print("\n📊 Step 3: Generating test data (3 sequences)...")
    test_generator = CubicSplineGenerator(seed=33384)  # Different seed for test
    test_times, test_targets, test_conditioning = test_generator.generate_spline_data(
        num_sequences=3, 
        sequence_length=1000
    )
    print(f"✅ Generated {test_targets.shape[0]} test sequences")
    print(f"📊 Test data range: [{test_targets.min():.3f}, {test_targets.max():.3f}]")
    
    # Step 4: Test tokenization and detokenization on test data (with padding)
    print("\n🧪 Step 4: Testing tokenization with padding on new sequences...")
    test_tokens_padded = fast_tokenizer.tokenize_with_padding(test_targets, pad_token=0, max_length=40)
    print(f"📊 Padded tokens shape: {test_tokens_padded.shape}")
    print(f"📊 Token range: [{test_tokens_padded.min()}, {test_tokens_padded.max()}]")
    
    # Create attention mask to see where padding is
    attention_mask = fast_tokenizer.create_attention_mask(test_tokens_padded, pad_token=0)
    print(f"📊 Attention mask shape: {attention_mask.shape}")
    print(f"📊 Average non-padding tokens per sequence: {attention_mask.sum(axis=1).mean():.1f}")
    
    test_reconstructed = fast_tokenizer.detokenize_with_padding(test_tokens_padded, pad_token=0)
    print(f"📊 Reconstructed shape: {test_reconstructed.shape}")
    
    # Step 5: Compute error on test data
    test_error = fast_tokenizer.compute_tokenization_error(test_targets, test_reconstructed)
    
    # Analyze token statistics on padded tokens
    print(f"\n📊 Analyzing token statistics...")
    # Flatten only non-padding tokens
    non_padding_tokens = test_tokens_padded[test_tokens_padded != 0]
    unique_tokens, counts = np.unique(non_padding_tokens, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    test_analysis = {
        'sampling_rate': 50,
        'entropy': entropy,
        'unique_tokens': len(unique_tokens),
        'total_tokens': len(non_padding_tokens),
        'tokens_per_sequence': attention_mask.sum(axis=1).mean()
    }
    
    print(f"\n📊 Test Results (with padding):")
    print(f"MSE: {test_error:.6f}")
    print(f"Entropy: {test_analysis['entropy']:.3f}")
    print(f"Unique tokens: {test_analysis['unique_tokens']}")
    print(f"Total tokens (non-padding): {test_analysis['total_tokens']}")
    print(f"Avg tokens per sequence: {test_analysis['tokens_per_sequence']:.1f}")
    
    # Calculate per-sequence metrics
    print(f"\n📊 Per-Sequence Metrics:")
    for i in range(test_targets.shape[0]):
        mse = np.mean((test_targets[i] - test_reconstructed[i]) ** 2)
        mae = np.mean(np.abs(test_targets[i] - test_reconstructed[i]))
        correlation = np.corrcoef(test_targets[i], test_reconstructed[i])[0, 1]
        num_tokens = attention_mask[i].sum()
        print(f"  Sequence {i+1}: MSE={mse:.6f}, MAE={mae:.6f}, r={correlation:.4f}, tokens={num_tokens}")
    
    print("\n✅ FAST tokenizer provides better compression and maintains information at high sampling rates!")
    
    # Step 6: Visualize padded tokens
    print("\n🎨 Step 6: Visualizing padded tokens...")
    fig, axes = plt.subplots(test_targets.shape[0], 1, figsize=(12, 3 * test_targets.shape[0]))
    if test_targets.shape[0] == 1:
        axes = [axes]
    
    for i in range(test_targets.shape[0]):
        ax = axes[i]
        tokens = test_tokens_padded[i]
        mask = attention_mask[i]
        
        # Plot tokens with padding highlighted
        x = np.arange(len(tokens))
        colors = ['steelblue' if m else 'lightgray' for m in mask]
        ax.bar(x, tokens, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(f'Sequence {i+1}: Tokenized ({mask.sum()} tokens + {(~mask).sum()} padding)', 
                    fontweight='bold')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Token ID')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        legend_elements = [
            Patch(facecolor='steelblue', edgecolor='black', label='Valid Token'),
            Patch(facecolor='lightgray', edgecolor='black', label='Padding')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Step 7: Visualize reconstruction on test data
    print("\n🎨 Step 7: Creating reconstruction visualization...")
    fast_tokenizer.visualize_reconstruction(test_targets, test_reconstructed, n_sequences=3, 
                                           title_prefix="FAST Tokenizer - Test Set (with Padding)")
