# Toy Problem Case Study: Tokenization Effects in VLA Training

This folder contains the implementation of the case study from the paper that demonstrates how naive tokenization schemes affect the training of autoregressive vision-language-action (VLA) policies at different sampling rates.

## Overview

The case study reproduces the key findings from the paper showing that:

1. **Naive binning tokenization** works well at low sampling rates (H=25-50)
2. **Performance degrades significantly** at high sampling rates (H=400-800)
3. **Marginal information content approaches zero** as sampling frequency increases
4. **Models tend to copy the first action** at high frequencies instead of learning meaningful patterns

This demonstrates the need for better tokenization schemes like the DCT-based FAST tokenization proposed in the paper.

## Files

- `cubic_spline_generator.py`: Generates synthetic cubic spline datasets with conditioning points
- `binning_tokenizer.py`: Implements naive binning tokenization (256 bins per element)
- `transformer_model.py`: Small autoregressive transformer model for sequence prediction
- `training.py`: Training loop and experiment runner
- `visualization.py`: Visualization utilities to replicate Figure 3 from the paper
- `main.py`: Main script to run the complete experiment
- `requirements.txt`: Python dependencies

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run a quick test:
```bash
python main.py --quick-test
```

3. Run the full experiment:
```bash
python main.py --sampling-rates 25 50 100 200 400 800 --num-sequences 1000 --num-epochs 100
```

4. Visualize existing results:
```bash
python main.py --skip-training
```

## Expected Results

The experiment should show:

- **Low MSE** (good performance) at sampling rates H=25, 50
- **Increasing MSE** (poor performance) as sampling rate increases
- **High MSE** at H=400, 800 where the model fails to learn meaningful patterns

This replicates the "naive" curve shown in Figure 3 of the paper.

## Key Insights

1. **Marginal Information Problem**: As sampling rate increases, consecutive tokens become highly correlated, reducing the marginal information content that autoregressive models rely on for learning.

2. **Token Redundancy**: At high frequencies, many consecutive tokens are identical or very similar, making it hard for the model to learn meaningful patterns.

3. **Copy Behavior**: Models trained at high sampling rates tend to simply copy the first action instead of interpolating the smooth spline curve.

4. **Need for Better Tokenization**: This case study motivates the development of better tokenization schemes like FAST that maintain high information content across all sampling rates.

## Visualization

The experiment generates several plots:

- `performance_vs_sampling_rate.png`: MSE vs sampling rate (replicates Figure 3 top)
- `marginal_information_analysis.png`: Shows entropy and token redundancy analysis
- `predictions_H*.png`: Prediction examples for different sampling rates
- `figure3_replica.png`: Comprehensive replica of Figure 3 from the paper

## Parameters

- `--sampling-rates`: List of sampling rates to test (default: [25, 50, 100, 200, 400, 800])
- `--num-sequences`: Number of sequences to generate (default: 1000)
- `--num-epochs`: Number of training epochs (default: 100)
- `--results-dir`: Directory to save results (default: 'results')
- `--quick-test`: Run with reduced parameters for faster testing
- `--skip-training`: Skip training and only visualize existing results

## Implementation Details

The implementation closely follows the paper's description:

1. **Cubic Spline Generation**: Creates smooth curves interpolating 4 random conditioning points
2. **Naive Tokenization**: Discretizes each element separately into 256 bins
3. **Autoregressive Training**: Trains transformer to predict next token given previous tokens
4. **Evaluation**: Measures MSE on continuous values after detokenization

This provides a controlled environment to study tokenization effects without the complexity of real robot data.
