"""
Visualization utilities for the Toy Problem Case Study

This module provides visualization functions to replicate Figure 3 from the paper,
showing the effect of sampling rate on prediction performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch

from cubic_spline_generator import CubicSplineGenerator
from fast_tokenizer import FASTTokenizer
from transformer_model import SimpleTransformer


class CaseStudyVisualizer:
    """Visualizer for the tokenization case study."""
    
    def __init__(self, device: torch.device = None):
        """Initialize the visualizer."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
    
    def plot_prediction_performance(self, 
                                   results: Dict[int, float],
                                   save_path: str = None) -> None:
        """
        Plot the prediction performance vs sampling rate (replicating Figure 3 top).
        
        Args:
            results: Dictionary mapping sampling rate to MSE
            save_path: Path to save the plot
        """
        sampling_rates = sorted(results.keys())
        mse_values = [results[H] for H in sampling_rates]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sampling_rates, mse_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Sampling Rate (H)')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Effect of Sampling Rate on Prediction Performance\n(FAST Tokenization)')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        
        # Add annotations
        for H, mse in zip(sampling_rates, mse_values):
            plt.annotate(f'{mse:.4f}', (H, mse), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_predictions(self, 
                             model: SimpleTransformer,
                             tokenizer: FASTTokenizer,
                             sampling_rate: int,
                             num_examples: int = 3,
                             save_path: str = None) -> None:
        """
        Visualize predictions for a specific sampling rate (replicating Figure 3 bottom).
        
        Args:
            model: Trained transformer model
            tokenizer: Fitted tokenizer
            sampling_rate: The sampling rate to visualize
            num_examples: Number of examples to show
            save_path: Path to save the plot
        """
        model.eval()
        
        # Generate test data
        generator = CubicSplineGenerator(seed=123)  # Different seed for variety
        times, targets, conditioning = generator.generate_spline_data(
            num_sequences=num_examples,
            sequence_length=sampling_rate
        )
        
        # Generate predictions
        with torch.no_grad():
            conditioning_tensor = torch.from_numpy(conditioning).float().to(self.device)
            predicted_tokens = model.generate(
                conditioning_tensor,
                max_length=sampling_rate,
                temperature=0.0,  # Deterministic
                device=self.device
            )
        
        # Convert predictions back to continuous values
        predicted_values = tokenizer.detokenize(predicted_tokens.cpu().numpy())
        
        # Create subplots
        fig, axes = plt.subplots(1, num_examples, figsize=(5 * num_examples, 5))
        if num_examples == 1:
            axes = [axes]
        
        for i in range(num_examples):
            ax = axes[i]
            
            # Plot ground truth
            ax.plot(times[i], targets[i], 'k--', linewidth=2, label='Ground Truth')
            
            # Plot conditioning points
            ax.scatter(conditioning[i, :, 0], conditioning[i, :, 1], 
                      color='white', s=100, zorder=5, edgecolors='black', linewidth=2,
                      label='Conditioning Points')
            
            # Plot prediction
            ax.plot(times[i], predicted_values[i], 'r-', linewidth=2, label='Prediction')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.set_title(f'Example {i+1} (H={sampling_rate})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Prediction Quality at Sampling Rate H={sampling_rate}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_marginal_information_analysis(self, 
                                          sampling_rates: List[int],
                                          save_path: str = None) -> None:
        """
        Plot the marginal information analysis for different sampling rates.
        
        Args:
            sampling_rates: List of sampling rates to analyze
            save_path: Path to save the plot
        """
        tokenizer = FASTTokenizer()
        generator = CubicSplineGenerator(seed=42)
        
        entropies = []
        zero_diff_ratios = []
        
        for H in sampling_rates:
            # Generate data
            times, targets, conditioning = generator.generate_spline_data(
                num_sequences=100,
                sequence_length=H
            )
            
            # Fit tokenizer
            tokenizer.fit(targets)
            
            # Analyze marginal information
            analysis = tokenizer.analyze_marginal_information(targets, H)
            entropies.append(analysis['entropy'])
            zero_diff_ratios.append(analysis['zero_diff_ratio'])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot entropy
        ax1.plot(sampling_rates, entropies, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Sampling Rate (H)')
        ax1.set_ylabel('Entropy of Token Differences')
        ax1.set_title('Marginal Information Content')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot zero difference ratio
        ax2.plot(sampling_rates, zero_diff_ratios, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Sampling Rate (H)')
        ax2.set_ylabel('Ratio of Zero Differences')
        ax2.set_title('Token Redundancy')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_figure3_replica(self, 
                              results: Dict[int, float],
                              model_paths: Dict[int, str],
                              save_path: str = None) -> None:
        """
        Create a replica of Figure 3 from the paper.
        
        Args:
            results: Dictionary mapping sampling rate to MSE
            model_paths: Dictionary mapping sampling rate to model path
            save_path: Path to save the plot
        """
        sampling_rates = sorted(results.keys())
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Top plot: MSE vs Sampling Rate
        ax1 = plt.subplot(2, 2, 1)
        mse_values = [results[H] for H in sampling_rates]
        ax1.plot(sampling_rates, mse_values, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Sampling Rate (H)')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('Prediction Performance vs Sampling Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Bottom plots: Prediction examples for different sampling rates
        example_rates = [25, 200, 800]  # Low, medium, high
        
        for i, H in enumerate(example_rates):
            ax = plt.subplot(2, 3, 4 + i)
            
            # Load model if available
            if H in model_paths and model_paths[H] is not None:
                model = SimpleTransformer(
                    vocab_size=256,
                    d_model=128,
                    nhead=8,
                    num_layers=4,
                    max_seq_len=H + 100
                )
                model.load_state_dict(torch.load(model_paths[H], map_location=self.device))
                model = model.to(self.device)
                
                # Generate prediction
                generator = CubicSplineGenerator(seed=42)
                times, targets, conditioning = generator.generate_spline_data(
                    num_sequences=1,
                    sequence_length=H
                )
                
                tokenizer = FASTTokenizer()
                tokenizer.fit(targets)
                
                with torch.no_grad():
                    conditioning_tensor = torch.from_numpy(conditioning).float().to(self.device)
                    predicted_tokens = model.generate(
                        conditioning_tensor,
                        max_length=H,
                        temperature=0.0,
                        device=self.device
                    )
                
                predicted_values = tokenizer.detokenize(predicted_tokens.cpu().numpy())
                
                # Plot
                ax.plot(times[0], targets[0], 'k--', linewidth=2, label='Ground Truth')
                ax.scatter(conditioning[0, :, 0], conditioning[0, :, 1], 
                          color='white', s=100, zorder=5, edgecolors='black', linewidth=2)
                ax.plot(times[0], predicted_values[0], 'r-', linewidth=2, label='Prediction')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.set_title(f'H = {H}')
                ax.grid(True, alpha=0.3)
                
                # Add MSE annotation
                mse = np.mean((targets[0] - predicted_values[0]) ** 2)
                ax.text(0.05, 0.95, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Test the visualizer
    visualizer = CaseStudyVisualizer()
    
    # Generate some dummy results for testing
    dummy_results = {25: 0.001, 50: 0.005, 100: 0.02, 200: 0.08, 400: 0.25, 800: 0.5}
    
    # Test plotting
    visualizer.plot_prediction_performance(dummy_results)
    visualizer.plot_marginal_information_analysis([25, 50, 100, 200, 400, 800])
