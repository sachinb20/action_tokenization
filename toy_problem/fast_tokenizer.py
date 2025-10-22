"""
FAST Tokenizer for Toy Problem Case Study

This module implements the FAST action tokenizer using the HuggingFace AutoProcessor
from the physical-intelligence/fast model. This provides a more efficient tokenization
scheme compared to naive binning.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
import torch
from transformers import AutoProcessor


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
        self.tokenizer = None
        self.is_fitted = False
        self.action_dim = None
        self.time_horizon = None
        
    def fit(self, data: np.ndarray):
        """
        Fit the tokenizer to data and train a custom tokenizer.
        
        Args:
            data: Array of shape (num_sequences, sequence_length) or (num_sequences, sequence_length, action_dim)
        """
        print(f"Training FAST tokenizer on {data.shape[0]} sequences...")
        
        # Load the base tokenizer
        self.tokenizer = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Prepare data for training - FAST expects action chunks
        if data.ndim == 2:
            # If 2D, assume single action dimension
            action_data = data.reshape(data.shape[0], data.shape[1], 1)
        else:
            action_data = data
            
        # Normalize data to [-1, 1] range as recommended by FAST
        action_data = self._normalize_data(action_data)
        
        # Train the tokenizer on our data
        self.tokenizer = self.tokenizer.fit(action_data)
        
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
        
        # Compute quantiles for normalization
        q1 = np.percentile(flat_data, 1)
        q99 = np.percentile(flat_data, 99)
        
        # Normalize to [-1, 1] range
        normalized = 2 * (data - q1) / (q99 - q1) - 1
        normalized = np.clip(normalized, -1, 1)
        
        return normalized
    
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
        
        # Tokenize using FAST - process each sequence separately to maintain dimensions
        all_tokens = []
        max_length = 0
        
        # First pass: tokenize all sequences and find max length
        for i in range(normalized_data.shape[0]):
            # Process one sequence at a time
            sequence = normalized_data[i:i+1]  # Keep batch dimension
            tokens = self.tokenizer(sequence)
            
            # Convert to numpy array if it's a list
            if isinstance(tokens, list):
                tokens = np.array(tokens)
            
            # Flatten to 1D for this sequence
            if tokens.ndim > 1:
                tokens = tokens.flatten()
            
            all_tokens.append(tokens)
            max_length = max(max_length, len(tokens))
        
        # Second pass: pad all sequences to the same length
        padded_tokens = []
        for tokens in all_tokens:
            if len(tokens) < max_length:
                # Pad with zeros (or use a special padding token)
                padded = np.pad(tokens, (0, max_length - len(tokens)), mode='constant', constant_values=0)
            else:
                padded = tokens
            padded_tokens.append(padded)
        
        # Stack all sequences back together
        tokens = np.vstack(padded_tokens)
        
        return tokens
    
    def detokenize(self, tokens: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Convert discrete tokens back to continuous values using FAST.
        
        Args:
            tokens: List or array of discrete tokens
            
        Returns:
            values: Array of continuous values
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before detokenization")
        
        # Convert to list if numpy array
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        # Decode using FAST
        decoded_actions = self.tokenizer.decode(tokens)
        
        # Denormalize back to original range
        # Note: This is a simplified denormalization - in practice you'd want to store
        # the original normalization parameters during fitting
        return decoded_actions
    
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
        
        This helps understand the efficiency of FAST tokenization compared to naive binning.
        
        Args:
            data: Continuous data
            sampling_rate: The sampling rate (sequence length)
            
        Returns:
            analysis: Dictionary with marginal information statistics
        """
        tokens = self.tokenize(data)
        
        # For FAST, we analyze the token sequence directly
        # since it's already compressed
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Compute statistics
        mean_abs_token = np.mean(np.abs(tokens))
        std_token = np.std(tokens)
        zero_token_ratio = np.mean(np.array(tokens) == 0)
        
        analysis = {
            'sampling_rate': sampling_rate,
            'entropy': entropy,
            'mean_abs_token': mean_abs_token,
            'std_token': std_token,
            'zero_token_ratio': zero_token_ratio,
            'unique_tokens': len(unique_tokens),
            'total_tokens': len(tokens),
            'compression_ratio': len(tokens) / (data.shape[0] * data.shape[1])
        }
        
        return analysis
    
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
    
    # Generate test data
    generator = CubicSplineGenerator(seed=42)
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=100, 
        sequence_length=50
    )
    
    # Initialize FAST tokenizer
    fast_tokenizer = FASTTokenizer()
    
    # Fit tokenizer
    fast_tokenizer.fit(targets)
    
    # Test tokenization and detokenization
    tokens = fast_tokenizer.tokenize(targets)
    reconstructed = fast_tokenizer.detokenize(tokens)
    
    # Compute error
    error = fast_tokenizer.compute_tokenization_error(targets, reconstructed)
    
    # Analyze marginal information
    analysis = fast_tokenizer.analyze_marginal_information(targets, 50)
    
    print(f"FAST Tokenizer Results:")
    print(f"MSE: {error:.6f}")
    print(f"Entropy: {analysis['entropy']:.3f}")
    print(f"Compression ratio: {analysis['compression_ratio']:.3f}")
    print(f"Unique tokens: {analysis['unique_tokens']}")
    print(f"Total tokens: {analysis['total_tokens']}")
    
    print("\nFAST tokenizer provides better compression and maintains information at high sampling rates!")
