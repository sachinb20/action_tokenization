"""
Cubic Spline Generator for Toy Problem Case Study

This module generates synthetic time series datasets where the goal is to predict
a cubic spline that interpolates four randomly-generated points.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from typing import Tuple, List


class CubicSplineGenerator:
    """Generates cubic spline datasets for the tokenization case study."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        self.rng = np.random.RandomState(seed)
    
    def generate_spline_data(self, 
                           num_sequences: int = 1000,
                           sequence_length: int = 100,
                           time_range: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate cubic spline data with conditioning points.
        
        Args:
            num_sequences: Number of sequences to generate
            sequence_length: Length of each sequence (sampling rate H)
            time_range: Time range for the spline (start, end)
            
        Returns:
            times: Array of shape (num_sequences, sequence_length) with time points
            targets: Array of shape (num_sequences, sequence_length) with target values
            conditioning_points: Array of shape (num_sequences, 4, 2) with (time, value) pairs
        """
        times = np.linspace(time_range[0], time_range[1], sequence_length)
        all_times = np.tile(times, (num_sequences, 1))
        all_targets = np.zeros((num_sequences, sequence_length))
        all_conditioning = np.zeros((num_sequences, 4, 2))
        
        for i in range(num_sequences):
            # Generate 4 random conditioning points
            condition_times = np.sort(self.rng.uniform(time_range[0], time_range[1], 4))
            condition_values = self.rng.uniform(-2, 2, 4)
            
            # Create cubic spline
            spline = CubicSpline(condition_times, condition_values, bc_type='natural')
            
            # Evaluate spline at all time points
            target_values = spline(times)
            
            all_targets[i] = target_values
            all_conditioning[i, :, 0] = condition_times
            all_conditioning[i, :, 1] = condition_values
        
        return all_times, all_targets, all_conditioning
    
    def visualize_example(self, 
                         sequence_length: int = 100,
                         time_range: Tuple[float, float] = (0.0, 1.0),
                         save_path: str = None):
        """Generate and visualize a single example spline."""
        times = np.linspace(time_range[0], time_range[1], sequence_length)
        
        # Generate conditioning points
        condition_times = np.sort(self.rng.uniform(time_range[0], time_range[1], 4))
        condition_values = self.rng.uniform(-2, 2, 4)
        
        # Create and evaluate spline
        spline = CubicSpline(condition_times, condition_values, bc_type='natural')
        target_values = spline(times)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(times, target_values, 'b-', linewidth=2, label='Cubic Spline')
        plt.scatter(condition_times, condition_values, color='red', s=100, zorder=5, label='Conditioning Points')
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Cubic Spline Example (H={sequence_length})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return times, target_values, condition_times, condition_values


if __name__ == "__main__":
    # Test the generator
    generator = CubicSplineGenerator(seed=42)
    
    # Generate a small dataset
    times, targets, conditioning = generator.generate_spline_data(
        num_sequences=100, 
        sequence_length=50
    )
    
    print(f"Generated dataset shape: {times.shape}")
    print(f"Target values range: [{targets.min():.3f}, {targets.max():.3f}]")
    
    # Visualize an example
    generator.visualize_example(sequence_length=100)
