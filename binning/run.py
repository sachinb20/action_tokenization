import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List

# Set up matplotlib for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Import our modules
from cubic_spline_generator import CubicSplineGenerator
from binning_tokenizer import BinningTokenizer
from transformer_model import SimpleTransformer, count_parameters
from training import Trainer, run_experiment
from visualization import CaseStudyVisualizer

print("Setup complete!")



# Initialize components
generator = CubicSplineGenerator(seed=42)
tokenizer = BinningTokenizer(num_bins=256)

# Test different sampling rates
sampling_rates = [25, 50, 100, 200, 400, 800]

print("Analyzing marginal information content:")
print("Sampling Rate | Entropy | Zero Diff Ratio | Unique Diffs")
print("-" * 60)

results = {}

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
    results[H] = analysis

    print(f"{H:13d} | {analysis['entropy']:7.3f} | {analysis['zero_diff_ratio']:13.3f} | {analysis['unique_diffs']:11d}")



# Set device
device = torch.device("cuda")
# if torch.cuda.is_available() else "cpu"


print(f"Using device: {device}")

# Create results directory
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Initialize visualizer
visualizer = CaseStudyVisualizer(device)

print("Setup complete for training experiment!")



# Run quick experiment
print("Running quick experiment with reduced parameters...")
print("Sampling rates: [25, 100, 400]")
print("Sequences: 200, Epochs: 20")

quick_results = run_experiment(
    sampling_rates=[25],
    num_sequences=10000,  # Smaller for faster testing
    num_epochs=200,      # Fewer epochs for faster testing
    results_dir=results_dir
)

print("\n" + "=" * 60)
print("QUICK EXPERIMENT RESULTS")
print("=" * 60)
print("Sampling Rate (H) | MSE")
print("-" * 30)
for H in sorted(quick_results.keys()):
    print(f"{H:15d} | {quick_results[H]:.6f}")
