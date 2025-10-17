"""
Training Loop for Toy Problem Case Study

This module implements the training loop for the autoregressive transformer
on the cubic spline prediction task with different sampling rates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import os

from cubic_spline_generator import CubicSplineGenerator
from binning_tokenizer import BinningTokenizer
from transformer_model import SimpleTransformer


class Trainer:
    """Trainer for the toy problem case study."""
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: BinningTokenizer,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        """
        Initialize the trainer.
        
        Args:
            model: The transformer model to train
            tokenizer: The tokenizer for converting data
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def prepare_data(self, 
                    times: np.ndarray,
                    targets: np.ndarray,
                    conditioning: np.ndarray,
                    train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            times: Time sequences
            targets: Target values
            conditioning: Conditioning points
            train_ratio: Ratio of data to use for training
            
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        num_sequences = len(times)
        train_size = int(num_sequences * train_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(num_sequences)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split data
        train_times = times[train_indices]
        train_targets = targets[train_indices]
        train_conditioning = conditioning[train_indices]
        
        val_times = times[val_indices]
        val_targets = targets[val_indices]
        val_conditioning = conditioning[val_indices]
        
        # Tokenize targets
        train_tokens = self.tokenizer.tokenize(train_targets)
        val_tokens = self.tokenizer.tokenize(val_targets)
        
        # Convert to tensors
        train_tokens = torch.from_numpy(train_tokens).long()
        train_conditioning = torch.from_numpy(train_conditioning).float()
        
        val_tokens = torch.from_numpy(val_tokens).long()
        val_conditioning = torch.from_numpy(val_conditioning).float()
        
        # Create datasets
        train_dataset = TensorDataset(train_tokens, train_conditioning)
        val_dataset = TensorDataset(val_tokens, val_conditioning)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_tokens, batch_conditioning in train_loader:
            batch_tokens = batch_tokens.to(self.device)
            batch_conditioning = batch_conditioning.to(self.device)
            
            # Create input and target sequences
            input_tokens = batch_tokens[:, :-1]  # All but last token
            target_tokens = batch_tokens[:, 1:]  # All but first token
            
            # Forward pass
            logits = self.model(input_tokens, batch_conditioning)
            
            # Compute loss
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_tokens, batch_conditioning in val_loader:
                batch_tokens = batch_tokens.to(self.device)
                batch_conditioning = batch_conditioning.to(self.device)
                
                # Create input and target sequences
                input_tokens = batch_tokens[:, :-1]
                target_tokens = batch_tokens[:, 1:]
                
                # Forward pass
                logits = self.model(input_tokens, batch_conditioning)
                
                # Compute loss
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              patience: int = 10,
              save_path: str = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save the best model
            
        Returns:
            history: Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            epoch_time = time.time() - start_time
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}, Time = {epoch_time:.2f}s")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def evaluate_mse(self, 
                    times: np.ndarray,
                    targets: np.ndarray,
                    conditioning: np.ndarray) -> float:
        """
        Evaluate the model using MSE on the continuous values.
        
        Args:
            times: Time sequences
            targets: Target values
            conditioning: Conditioning points
            
        Returns:
            mse: Mean squared error
        """
        self.model.eval()
        
        # Tokenize targets
        target_tokens = self.tokenizer.tokenize(targets)
        
        # Convert to tensors
        target_tokens = torch.from_numpy(target_tokens).long().to(self.device)
        conditioning = torch.from_numpy(conditioning).float().to(self.device)
        
        # Generate predictions
        with torch.no_grad():
            predicted_tokens = self.model.generate(
                conditioning, 
                max_length=target_tokens.shape[1],
                temperature=0.0,  # Deterministic generation
                device=self.device
            )
        
        # Convert back to continuous values
        predicted_continuous = self.tokenizer.detokenize(
            predicted_tokens.cpu().numpy()
        )
        
        # Compute MSE
        mse = np.mean((targets - predicted_continuous) ** 2)
        
        return mse


def run_experiment(sampling_rates: List[int] = None,
                  num_sequences: int = 1000,
                  num_epochs: int = 100,
                  results_dir: str = "results") -> Dict[int, float]:
    """
    Run the complete experiment with different sampling rates.
    
    Args:
        sampling_rates: List of sampling rates to test
        num_sequences: Number of sequences to generate
        num_epochs: Number of training epochs
        results_dir: Directory to save results
        
    Returns:
        results: Dictionary mapping sampling rate to MSE
    """
    if sampling_rates is None:
        sampling_rates = [25, 50, 100, 200, 400, 800]
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    for H in sampling_rates:
        print(f"\n{'='*50}")
        print(f"Training with sampling rate H = {H}")
        print(f"{'='*50}")
        
        # Generate data
        generator = CubicSplineGenerator(seed=42)
        times, targets, conditioning = generator.generate_spline_data(
            num_sequences=num_sequences,
            sequence_length=H
        )
        
        # Initialize tokenizer and fit to data
        tokenizer = BinningTokenizer(num_bins=256)
        tokenizer.fit(targets)
        
        # Create model
        model = SimpleTransformer(
            vocab_size=256,
            d_model=128,
            nhead=8,
            num_layers=4,
            max_seq_len=H + 100  # Add some buffer
        )
        
        # Initialize trainer
        trainer = Trainer(model, tokenizer, device)
        
        # Prepare data
        train_loader, val_loader = trainer.prepare_data(times, targets, conditioning)
        
        # Train model
        model_path = os.path.join(results_dir, f"model_H{H}.pth")
        history = trainer.train(
            train_loader, 
            val_loader, 
            num_epochs=num_epochs,
            save_path=model_path
        )
        
        # Evaluate MSE
        mse = trainer.evaluate_mse(times, targets, conditioning)
        results[H] = mse
        
        print(f"H = {H}: MSE = {mse:.6f}")
        
        # Save training history
        history_path = os.path.join(results_dir, f"history_H{H}.npz")
        np.savez(history_path, **history)
        
        # Clean up for next iteration
        del model, trainer, train_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


if __name__ == "__main__":
    # Run the experiment
    print("Starting Toy Problem Case Study Experiment")
    print("=" * 60)
    
    results = run_experiment(
        sampling_rates=[25, 50, 100, 200, 400, 800],
        num_sequences=500,  # Smaller for faster testing
        num_epochs=50,      # Fewer epochs for faster testing
        results_dir="results"
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print("Sampling Rate (H) | MSE")
    print("-" * 30)
    for H in sorted(results.keys()):
        print(f"{H:15d} | {results[H]:.6f}")
    
    print("\nNote: As predicted by the paper, MSE should increase with sampling rate")
    print("due to the marginal information content approaching zero.")
