#!/usr/bin/env python3
"""
Debug utility to diagnose prediction issues.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from cubic_spline_generator import CubicSplineGenerator
from binning_tokenizer import BinningTokenizer
from fast_tokenizer import FASTTokenizer
from transformer_model import SimpleTransformer


def debug_single_prediction(model, tokenizer, conditioning, target_values, device='cpu'):
    """
    Debug a single prediction to see where things go wrong.
    """
    print("=== DEBUGGING SINGLE PREDICTION ===\n")
    
    # Tokenize target
    target_tokens = tokenizer.tokenize(target_values.reshape(1, -1))[0]
    print(f"Target tokens shape: {target_tokens.shape}")
    print(f"Target tokens range: [{target_tokens.min()}, {target_tokens.max()}]")
    print(f"Target tokens (first 10): {target_tokens[:10]}")
    print()
    
    # Convert to tensors
    conditioning_tensor = torch.from_numpy(conditioning.reshape(1, 4, 2)).float().to(device)
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        predicted_tokens = model.generate(
            conditioning_tensor,
            max_length=len(target_tokens),
            greedy=True,
            device=device
        )[0].cpu().numpy()
    
    print(f"Predicted tokens shape: {predicted_tokens.shape}")
    print(f"Predicted tokens range: [{predicted_tokens.min()}, {predicted_tokens.max()}]")
    print(f"Predicted tokens (first 10): {predicted_tokens[:10]}")
    print()
    
    # Detokenize
    predicted_values = tokenizer.detokenize(predicted_tokens.reshape(1, -1))[0]
    
    # Compute metrics
    mse = np.mean((target_values - predicted_values) ** 2)
    mae = np.mean(np.abs(target_values - predicted_values))
    
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print()
    
    # Token accuracy
    token_accuracy = np.mean(target_tokens == predicted_tokens)
    print(f"Token Accuracy: {token_accuracy:.2%}")
    print()
    
    # Visualize
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(target_values, 'b-', label='Ground Truth', linewidth=2)
    plt.plot(predicted_values, 'r--', label='Prediction', linewidth=2)
    plt.scatter(conditioning[:, 0] * len(target_values), conditioning[:, 1], 
                c='green', s=100, zorder=5, label='Conditioning Points')
    plt.legend()
    plt.title('Continuous Values Comparison')
    plt.ylabel('Value')
    
    plt.subplot(3, 1, 2)
    plt.plot(target_tokens, 'b-', label='Target Tokens', linewidth=2)
    plt.plot(predicted_tokens, 'r--', label='Predicted Tokens', linewidth=2)
    plt.legend()
    plt.title('Token Sequence Comparison')
    plt.ylabel('Token ID')
    
    plt.subplot(3, 1, 3)
    plt.plot(np.abs(target_values - predicted_values), 'r-', linewidth=2)
    plt.title('Absolute Error')
    plt.ylabel('|Error|')
    plt.xlabel('Position')
    
    plt.tight_layout()
    plt.savefig('debug_prediction.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to debug_prediction.png")
    
    return mse, mae, token_accuracy


def check_model_learning(model, train_loader, device='cpu'):
    """
    Check if the model is actually learning.
    """
    print("=== CHECKING MODEL LEARNING ===\n")
    
    model.eval()
    total_loss = 0
    total_token_acc = 0
    num_batches = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_tokens, batch_conditioning in train_loader:
            if num_batches >= 10:  # Only check first 10 batches
                break
                
            batch_tokens = batch_tokens.to(device)
            batch_conditioning = batch_conditioning.to(device)
            
            input_tokens = batch_tokens[:, :-1]
            target_tokens = batch_tokens[:, 1:]
            
            logits = model(input_tokens, batch_conditioning)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
            
            # Token accuracy
            predictions = torch.argmax(logits, dim=-1)
            token_acc = (predictions == target_tokens).float().mean()
            
            total_loss += loss.item()
            total_token_acc += token_acc.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_token_acc = total_token_acc / num_batches
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Token Accuracy: {avg_token_acc:.2%}")
    print()
    
    if avg_loss > 5.0:
        print("⚠️  WARNING: Loss is very high! Model may not be trained yet.")
    elif avg_token_acc < 0.1:
        print("⚠️  WARNING: Token accuracy is very low! Model is not learning.")
    elif avg_token_acc > 0.9:
        print("✅ Model seems to be learning well!")
    else:
        print("ℹ️  Model is learning but could be better.")
    
    return avg_loss, avg_token_acc


def check_tokenizer_quality(tokenizer, targets):
    """
    Check if the tokenizer is working correctly.
    """
    print("=== CHECKING TOKENIZER QUALITY ===\n")
    
    # Tokenize and detokenize
    tokens = tokenizer.tokenize(targets)
    reconstructed = tokenizer.detokenize(tokens)
    
    # Compute reconstruction error
    reconstruction_error = np.mean((targets - reconstructed) ** 2)
    
    print(f"Reconstruction MSE: {reconstruction_error:.6f}")
    print(f"Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"Num unique tokens: {len(np.unique(tokens))}")
    print()
    
    if reconstruction_error > 0.1:
        print("⚠️  WARNING: High reconstruction error! Tokenizer may be lossy.")
    else:
        print("✅ Tokenizer reconstruction is good.")
    
    return reconstruction_error


if __name__ == "__main__":
    print("This is a utility module. Import and use the debug functions.\n")
    print("Example usage:")
    print("  from debug_predictions import debug_single_prediction")
    print("  debug_single_prediction(model, tokenizer, conditioning, targets)")


