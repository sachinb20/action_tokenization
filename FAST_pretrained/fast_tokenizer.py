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
        Initialize the FAST tokenizer with pretrained weights.
        
        Args:
            model_name: HuggingFace model name for the FAST tokenizer
        """
        self.model_name = model_name
        print(f"Loading pretrained FAST tokenizer from '{model_name}'...")
        # Load the pretrained tokenizer directly
        self.tokenizer = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        print(f"Pretrained FAST tokenizer loaded successfully!")
    
    def tokenize(self, data: np.ndarray) -> List[np.ndarray]:
        """Convert continuous values to discrete tokens - exactly like test_pretrained_fast.py"""
        if data.ndim == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        
        all_tokens = []
        for i in range(data.shape[0]):
            sequence = data[i:i+1]  # (1, seq_len, 1)
            tokens = self.tokenizer(sequence)  # Returns list, exactly like test_pretrained_fast.py
            
            if isinstance(tokens, list) and len(tokens) > 0:
                tokens = np.array(tokens[0])
            else:
                tokens = np.array(tokens) if not isinstance(tokens, np.ndarray) else tokens
            
            if tokens.ndim > 1:
                tokens = tokens.flatten()
            
            all_tokens.append(tokens.astype(np.int64))
        
        return all_tokens
    
    def tokenize_with_padding(self, data: np.ndarray, pad_token: int = 0, max_length: int = 40, vocab_size: int = None) -> np.ndarray:
        """
        Convert continuous values to discrete tokens using FAST with padding for transformer compatibility.
        
        Args:
            data: Array of continuous values with shape (num_sequences, sequence_length) 
                  or (num_sequences, sequence_length, action_dim)
            pad_token: Token value to use for padding (default: 0)
            max_length: Maximum sequence length for padding (default: 40)
            vocab_size: Optional vocabulary size to clamp tokens to [0, vocab_size)
            
        Returns:
            tokens: Padded array of discrete tokens [batch_size, max_length]
        """
        # Get variable-length tokens first
        variable_tokens = self.tokenize(data)
        
        # Pad all sequences to max_length
        padded_tokens = []
        for tokens in variable_tokens:
            if vocab_size is not None:
                tokens = np.clip(tokens, 0, vocab_size - 1)
            
            if len(tokens) < max_length:
                padded = np.pad(tokens, (0, max_length - len(tokens)), 
                              mode='constant', constant_values=pad_token)
            else:
                padded = tokens[:max_length]
            
            padded_tokens.append(padded.astype(np.int64))
        
        return np.array(padded_tokens, dtype=np.int64)
    
    def detokenize(self, tokens: Union[List[int], np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Convert tokens back to continuous values - exactly like test_pretrained_fast.py"""
        if isinstance(tokens, list):
            if len(tokens) > 0 and isinstance(tokens[0], np.ndarray):
                tokens_list = [t.tolist() for t in tokens]
            elif len(tokens) > 0 and isinstance(tokens[0], (int, np.integer)):
                tokens_list = [tokens]
            else:
                tokens_list = tokens
        elif isinstance(tokens, np.ndarray):
            tokens_list = [tokens.tolist()] if tokens.ndim == 1 else [tokens[i].tolist() for i in range(tokens.shape[0])]
        else:
            raise ValueError(f"Unsupported token format: {type(tokens)}")
        
        decoded = self.tokenizer.decode(tokens_list)
        
        if isinstance(decoded, np.ndarray) and decoded.ndim == 3 and decoded.shape[-1] == 1:
            decoded = decoded.squeeze(-1)
        
        return decoded
    
    def detokenize_with_padding(self, padded_tokens: np.ndarray, pad_token: int = 0) -> np.ndarray:
        """
        Convert padded discrete tokens back to continuous values using FAST.
        
        Args:
            padded_tokens: Padded array of discrete tokens [batch_size, max_length] or [max_length]
            pad_token: Token value used for padding (default: 0)
            
        Returns:
            values: Array of continuous values
        """
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
    
    def save_pretrained(self, path: str):
        """
        Save the trained tokenizer.
        
        Args:
            path: Path to save the tokenizer
        """
        self.tokenizer.save_pretrained(path)
    
    def load_pretrained(self, path: str):
        """
        Load a pre-trained tokenizer.
        
        Args:
            path: Path to load the tokenizer from
        """
        self.tokenizer = AutoProcessor.from_pretrained(path, trust_remote_code=True)


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
    
    # Initialize FAST tokenizer (loads pretrained weights automatically)
    fast_tokenizer = FASTTokenizer()
    
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
