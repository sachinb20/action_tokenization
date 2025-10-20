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
                           time_range: Tuple[float, float] = (0.0, 1.0),
                           value_range: Tuple[float, float] = (-8.0, 8.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate cubic spline data with conditioning points.
        
        Args:
            num_sequences: Number of sequences to generate
            sequence_length: Length of each sequence (sampling rate H)
            time_range: Time range for the spline (start, end)
            value_range: Value range constraint (min, max) for the spline function
            
        Returns:
            times: Array of shape (num_sequences, sequence_length) with time points
            targets: Array of shape (num_sequences, sequence_length) with target values
            conditioning_points: Array of shape (num_sequences, 4, 2) with (time, value) pairs
        """
        times = np.linspace(time_range[0], time_range[1], sequence_length)
        all_times = np.tile(times, (num_sequences, 1))
        all_targets = np.zeros((num_sequences, sequence_length))
        all_conditioning = np.zeros((num_sequences, 4, 2))
        
        valid_count = 0
        max_attempts = num_sequences * 10  # Allow up to 10x attempts to find valid splines
        attempt_count = 0
        
        while valid_count < num_sequences and attempt_count < max_attempts:
            attempt_count += 1
            
            # Generate conditioning points with 2 fixed at start/end
            condition_times = np.zeros(4)
            condition_values = np.zeros(4)
            
            # Fix start and end points with constrained values
            condition_times[0] = time_range[0]  # Start point
            condition_times[3] = time_range[1]  # End point
            condition_values[0] = self.rng.uniform(2, 8)  # Constrained start value
            condition_values[3] = self.rng.uniform(2, 8)  # Constrained end value
            
            # Generate 2 random intermediate points in middle range
            margin = (time_range[1] - time_range[0]) * 0.1
            middle_start = time_range[0] + margin
            middle_end = time_range[1] - margin
            
            condition_times[1:3] = np.sort(self.rng.uniform(middle_start, middle_end, 2))
            condition_values[1:3] = self.rng.uniform(-8, 0, 2)  # Constrained intermediate values
            
            # Ensure times are sorted
            sort_indices = np.argsort(condition_times)
            condition_times = condition_times[sort_indices]
            condition_values = condition_values[sort_indices]
            
            # Create cubic spline
            spline = CubicSpline(condition_times, condition_values, bc_type='natural')
            
            # Evaluate spline at all time points
            target_values = spline(times)
            
            # Check if all values are within the specified range
            if np.all((target_values >= value_range[0]) & (target_values <= value_range[1])):
                all_targets[valid_count] = target_values
                all_conditioning[valid_count, :, 0] = condition_times
                all_conditioning[valid_count, :, 1] = condition_values
                valid_count += 1
        
        if valid_count < num_sequences:
            print(f"Warning: Only generated {valid_count}/{num_sequences} valid splines within range {value_range}")
            # Truncate arrays to actual valid count
            all_times = all_times[:valid_count]
            all_targets = all_targets[:valid_count]
            all_conditioning = all_conditioning[:valid_count]
        
        return all_times, all_targets, all_conditioning
    
    def visualize_example(self, 
                         sequence_length: int = 100,
                         time_range: Tuple[float, float] = (0.0, 1.0),
                         value_range: Tuple[float, float] = (-8.0, 8.0),
                         save_path: str = None):
        """Generate and visualize a single example spline."""
        times = np.linspace(time_range[0], time_range[1], sequence_length)
        
        # Keep trying until we get a valid spline within the range
        max_attempts = 100
        for attempt in range(max_attempts):
            # Generate conditioning points with 2 fixed at start/end
            condition_times = np.zeros(4)
            condition_values = np.zeros(4)
            
            # Fix start and end points with constrained values
            condition_times[0] = time_range[0]  # Start point
            condition_times[3] = time_range[1]  # End point
            condition_values[0] = self.rng.uniform(2, 8)  # Constrained start value
            condition_values[3] = self.rng.uniform(2, 8)  # Constrained end value
            
            # Generate 2 random intermediate points in middle range
            margin = (time_range[1] - time_range[0]) * 0.1
            middle_start = time_range[0] + margin
            middle_end = time_range[1] - margin
            
            condition_times[1:3] = np.sort(self.rng.uniform(middle_start, middle_end, 2))
            condition_values[1:3] = self.rng.uniform(-8, 0, 2)  # Constrained intermediate values
            
            # Ensure times are sorted
            sort_indices = np.argsort(condition_times)
            condition_times = condition_times[sort_indices]
            condition_values = condition_values[sort_indices]
            
            # Create and evaluate spline
            spline = CubicSpline(condition_times, condition_values, bc_type='natural')
            target_values = spline(times)
            
            # Check if all values are within the specified range
            if np.all((target_values >= value_range[0]) & (target_values <= value_range[1])):
                break
        else:
            print(f"Warning: Could not generate a valid spline within range {value_range} after {max_attempts} attempts")
            # Use the last generated spline anyway for visualization
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(times, target_values, 'b-', linewidth=2, label='Cubic Spline')
        plt.scatter(condition_times, condition_values, color='red', s=100, zorder=5, label='Conditioning Points')
        
        # Highlight start and end points
        plt.scatter(condition_times[0], condition_values[0], color='green', s=150, zorder=6, 
                   marker='s', label='Start Point')
        plt.scatter(condition_times[-1], condition_values[-1], color='orange', s=150, zorder=6, 
                   marker='^', label='End Point')
        
        # Add horizontal lines to show value range constraints
        plt.axhline(y=value_range[0], color='gray', linestyle='--', alpha=0.7, label=f'Min Value ({value_range[0]})')
        plt.axhline(y=value_range[1], color='gray', linestyle='--', alpha=0.7, label=f'Max Value ({value_range[1]})')
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Cubic Spline Example (H={sequence_length}) - Constrained to [{value_range[0]}, {value_range[1]}]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(value_range[0] - 1, value_range[1] + 1)  # Add small margin for visibility
        
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
    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)
    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)
    generator.visualize_example(sequence_length=100)
    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)

    generator.visualize_example(sequence_length=100)
