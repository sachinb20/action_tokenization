"""
Main script for the Toy Problem Case Study

This script reproduces the case study from the paper showing how tokenization
affects VLA training performance at different sampling rates.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, List

from cubic_spline_generator import CubicSplineGenerator
from binning_tokenizer import BinningTokenizer
from transformer_model import SimpleTransformer, count_parameters
from training import Trainer, run_experiment
from visualization import CaseStudyVisualizer


def main():
    """Main function to run the case study experiment."""
    parser = argparse.ArgumentParser(description='Toy Problem Case Study')
    parser.add_argument('--sampling-rates', nargs='+', type=int, 
                       default=[25, 50, 100, 200, 400, 800],
                       help='Sampling rates to test')
    parser.add_argument('--num-sequences', type=int, default=1000,
                       help='Number of sequences to generate')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only visualize existing results')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick test with smaller parameters')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick test
    if args.quick_test:
        args.sampling_rates = [25, 100, 400]
        args.num_sequences = 200
        args.num_epochs = 20
        print("Running quick test with reduced parameters...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = CaseStudyVisualizer(device)
    
    if not args.skip_training:
        print("=" * 60)
        print("TOY PROBLEM CASE STUDY: NAIVE BINNING TOKENIZATION")
        print("=" * 60)
        print("This experiment reproduces the case study from the paper")
        print("showing how naive tokenization fails at high sampling rates.")
        print("=" * 60)
        
        # Run the experiment
        results = run_experiment(
            sampling_rates=args.sampling_rates,
            num_sequences=args.num_sequences,
            num_epochs=args.num_epochs,
            results_dir=args.results_dir
        )
        
        # Save results
        results_path = os.path.join(args.results_dir, 'results.npz')
        np.savez(results_path, **{f'H_{H}': mse for H, mse in results.items()})
        
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)
        print("Sampling Rate (H) | MSE")
        print("-" * 30)
        for H in sorted(results.keys()):
            print(f"{H:15d} | {results[H]:.6f}")
    
    else:
        # Load existing results
        results_path = os.path.join(args.results_dir, 'results.npz')
        if os.path.exists(results_path):
            data = np.load(results_path)
            results = {int(k[2:]): float(v) for k, v in data.items()}  # Remove 'H_' prefix
            print("Loaded existing results:")
            for H in sorted(results.keys()):
                print(f"H={H}: MSE={results[H]:.6f}")
        else:
            print("No existing results found. Please run without --skip-training first.")
            return
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Performance vs Sampling Rate
    perf_path = os.path.join(args.results_dir, 'performance_vs_sampling_rate.png')
    visualizer.plot_prediction_performance(results, save_path=perf_path)
    
    # Plot 2: Marginal Information Analysis
    info_path = os.path.join(args.results_dir, 'marginal_information_analysis.png')
    visualizer.plot_marginal_information_analysis(args.sampling_rates, save_path=info_path)
    
    # Plot 3: Prediction Examples (if models are available)
    model_paths = {}
    for H in args.sampling_rates:
        model_path = os.path.join(args.results_dir, f'model_H{H}.pth')
        if os.path.exists(model_path):
            model_paths[H] = model_path
    
    if model_paths:
        # Visualize predictions for different sampling rates
        for H in [min(args.sampling_rates), max(args.sampling_rates)]:
            if H in model_paths:
                # Load model and tokenizer
                model = SimpleTransformer(
                    vocab_size=256,
                    d_model=128,
                    nhead=8,
                    num_layers=4,
                    max_seq_len=H + 100
                )
                model.load_state_dict(torch.load(model_paths[H], map_location=device))
                model = model.to(device)
                
                # Generate data and fit tokenizer
                generator = CubicSplineGenerator(seed=42)
                times, targets, conditioning = generator.generate_spline_data(
                    num_sequences=100, sequence_length=H
                )
                tokenizer = BinningTokenizer(num_bins=256)
                tokenizer.fit(targets)
                
                # Visualize predictions
                pred_path = os.path.join(args.results_dir, f'predictions_H{H}.png')
                visualizer.visualize_predictions(
                    model, tokenizer, H, num_examples=3, save_path=pred_path
                )
    
    # Create comprehensive Figure 3 replica
    if model_paths:
        fig3_path = os.path.join(args.results_dir, 'figure3_replica.png')
        visualizer.create_figure3_replica(results, model_paths, save_path=fig3_path)
    
    print(f"\nResults and visualizations saved to: {args.results_dir}")
    
    # Print analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("As predicted by the paper:")
    print("1. Low sampling rates (H=25-50): Good performance, low MSE")
    print("2. High sampling rates (H=400-800): Poor performance, high MSE")
    print("3. The model struggles to learn meaningful patterns at high frequencies")
    print("4. This is due to marginal information content approaching zero")
    print("5. The model tends to copy the first action at high sampling rates")
    print("\nThis demonstrates the need for better tokenization schemes")
    print("like the DCT-based FAST tokenization proposed in the paper.")


def demo_tokenization_issue():
    """Demonstrate the tokenization issue with a simple example."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING THE TOKENIZATION ISSUE")
    print("=" * 60)
    
    # Generate data at different sampling rates
    generator = CubicSplineGenerator(seed=42)
    tokenizer = BinningTokenizer(num_bins=256)
    
    sampling_rates = [25, 100, 400]
    
    print("Analyzing marginal information content:")
    print("Sampling Rate | Entropy | Zero Diff Ratio")
    print("-" * 45)
    
    for H in sampling_rates:
        # Generate data
        times, targets, conditioning = generator.generate_spline_data(
            num_sequences=100, sequence_length=H
        )
        
        # Fit tokenizer
        tokenizer.fit(targets)
        
        # Analyze marginal information
        analysis = tokenizer.analyze_marginal_information(targets, H)
        
        print(f"{H:13d} | {analysis['entropy']:7.3f} | {analysis['zero_diff_ratio']:13.3f}")
    
    print("\nKey observations:")
    print("- As sampling rate increases, entropy decreases")
    print("- Zero difference ratio increases (more redundant tokens)")
    print("- This makes learning much harder for autoregressive models")


if __name__ == "__main__":
    # Run demo first
    demo_tokenization_issue()
    
    # Run main experiment
    main()
