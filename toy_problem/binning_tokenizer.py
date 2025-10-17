"""
Naive Binning Tokenizer for Toy Problem Case Study

This module implements the naive tokenization scheme employed in previous VLA policies,
which discretizes each element in the sequence separately into one of 256 bins.
"""

import numpy as np
from typing import Tuple, Union


class BinningTokenizer:
    """
    Naive binning tokenizer that discretizes each element separately into 256 bins.
    
    This is the baseline tokenization approach that suffers from poor performance
    at high sampling rates due to the marginal information content approaching zero.
    """
    
    def __init__(self, num_bins: int = 256):
        """
        Initialize the tokenizer.
        
        Args:
            num_bins: Number of bins for discretization (default 256)
        """
        self.num_bins = num_bins
        self.value_range = None
        self.bin_width = None
    
    def fit(self, data: np.ndarray):
        """
        Fit the tokenizer to data to determine value range and bin width.
        
        Args:
            data: Array of shape (num_sequences, sequence_length) or (sequence_length,)
        """
        # Flatten data to find global min/max
        flat_data = data.flatten()
        min_val = flat_data.min()
        max_val = flat_data.max()
        
        # Add small padding to avoid edge cases
        padding = (max_val - min_val) * 0.01
        self.value_range = (min_val - padding, max_val + padding)
        self.bin_width = (self.value_range[1] - self.value_range[0]) / self.num_bins
        
        print(f"Fitted tokenizer with range [{self.value_range[0]:.3f}, {self.value_range[1]:.3f}]")
        print(f"Bin width: {self.bin_width:.3f}")
    
    def tokenize(self, data: np.ndarray) -> np.ndarray:
        """
        Convert continuous values to discrete tokens.
        
        Args:
            data: Array of continuous values
            
        Returns:
            tokens: Array of discrete tokens (0 to num_bins-1)
        """
        if self.value_range is None:
            raise ValueError("Tokenizer must be fitted before tokenization")
        
        # Normalize to [0, num_bins)
        normalized = (data - self.value_range[0]) / (self.value_range[1] - self.value_range[0])
        tokens = np.floor(normalized * self.num_bins).astype(np.int32)
        
        # Clamp to valid range
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        
        return tokens
    
    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert discrete tokens back to continuous values.
        
        Args:
            tokens: Array of discrete tokens
            
        Returns:
            values: Array of continuous values
        """
        if self.value_range is None:
            raise ValueError("Tokenizer must be fitted before detokenization")
        
        # Convert tokens back to normalized values
        normalized = (tokens + 0.5) / self.num_bins  # Add 0.5 for center of bin
        
        # Denormalize to original range
        values = normalized * (self.value_range[1] - self.value_range[0]) + self.value_range[0]
        
        return values
    
    def compute_tokenization_error(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Compute the MSE between original and reconstructed data.
        
        Args:
            original: Original continuous data
            reconstructed: Reconstructed data from tokens
            
        Returns:
            mse: Mean squared error
        """
        return np.mean((original - reconstructed) ** 2)
    
    def analyze_marginal_information(self, data: np.ndarray, sampling_rate: int) -> dict:
        """
        Analyze the marginal information content for different sampling rates.
        
        This helps understand why naive tokenization fails at high frequencies.
        
        Args:
            data: Continuous data
            sampling_rate: The sampling rate (sequence length)
            
        Returns:
            analysis: Dictionary with marginal information statistics
        """
        tokens = self.tokenize(data)
        
        # Compute differences between consecutive tokens
        token_diffs = np.diff(tokens, axis=-1)
        
        # Compute marginal information (entropy of differences)
        unique_diffs, counts = np.unique(token_diffs.flatten(), return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Compute statistics
        mean_abs_diff = np.mean(np.abs(token_diffs))
        std_diff = np.std(token_diffs)
        zero_diff_ratio = np.mean(token_diffs == 0)
        
        analysis = {
            'sampling_rate': sampling_rate,
            'entropy': entropy,
            'mean_abs_diff': mean_abs_diff,
            'std_diff': std_diff,
            'zero_diff_ratio': zero_diff_ratio,
            'unique_diffs': len(unique_diffs),
            'total_possible_diffs': 2 * self.num_bins - 1  # -num_bins to +num_bins
        }
        
        return analysis


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = BinningTokenizer(num_bins=256)
    
    # Generate test data
    from cubic_spline_generator import CubicSplineGenerator
    generator = CubicSplineGenerator(seed=42)
    
    # Test with different sampling rates
    sampling_rates = [25, 50, 100, 200, 400, 800]
    
    print("Testing Binning Tokenizer:")
    print("=" * 50)
    
    for H in sampling_rates:
        # Generate data
        times, targets, conditioning = generator.generate_spline_data(
            num_sequences=100, 
            sequence_length=H
        )
        
        # Fit tokenizer
        tokenizer.fit(targets)
        
        # Tokenize and detokenize
        tokens = tokenizer.tokenize(targets)
        reconstructed = tokenizer.detokenize(tokens)
        
        # Compute error
        error = tokenizer.compute_tokenization_error(targets, reconstructed)
        
        # Analyze marginal information
        analysis = tokenizer.analyze_marginal_information(targets, H)
        
        print(f"H={H:3d}: MSE={error:.6f}, "
              f"Entropy={analysis['entropy']:.3f}, "
              f"Zero_diff_ratio={analysis['zero_diff_ratio']:.3f}")
    
    print("\nNote: As sampling rate increases, marginal information decreases")
    print("This explains why naive tokenization fails at high frequencies.")
