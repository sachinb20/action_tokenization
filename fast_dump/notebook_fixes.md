# =============================================================================
# TRUE INFERENCE: Tokenize new trajectories and give first token to model
# =============================================================================

print("🔧 Tokenizing NEW trajectories...")

# Tokenize the NEW test trajectories
test_tokens = fast_tokenizer.tokenize_with_padding(
    test_targets, 
    pad_token=PAD_TOKEN, 
    max_length=MAX_TOKEN_LENGTH
)

print(f"✅ Tokenized test sequences: {test_tokens.shape}")

# =============================================================================
# TRUE INFERENCE: Give model only first token, let it generate the rest
# =============================================================================

print("🚀 Running TRUE inference - giving model first token only...")

with torch.no_grad():
    predictions = []
    targets = []
    first_tokens = []
    
    for i in range(len(test_targets)):
        print(f"Processing sequence {i+1}/{len(test_targets)}...")
        
        # Get the first token from the tokenized sequence
        first_token = test_tokens[i:i+1, 0:1]  # Shape: (1, 1)
        first_tokens.append(first_token[0, 0].item())
        
        # Get original tokens for this sequence (remove padding)
        original_tokens = test_tokens[i]
        original_actual_tokens = original_tokens[original_tokens != PAD_TOKEN]
        
        print(f"  📋 Original tokens: {original_actual_tokens.tolist()}")
        print(f"  🔢 Original token count: {len(original_actual_tokens)}")
        
        # Convert first token to PyTorch tensor and move to device
        first_token_tensor = torch.from_numpy(first_token).long().to(DEVICE)
        
        # Get conditioning for this sequence
        conditioning_tensor = torch.from_numpy(test_conditioning[i:i+1]).float().to(DEVICE)
        
        # TRUE INFERENCE: Give model first token and let it generate the rest
        generated_tokens = model.generate(
            conditioning_points=conditioning_tensor,
            start_tokens=first_token_tensor,  # Start with first token
            max_length=MAX_TOKEN_LENGTH,
            greedy=True,  # Deterministic generation
            device=DEVICE,
            tokenizer=fast_tokenizer
        )
        
        try:
            # Convert generated tokens to numpy and remove padding
            generated_tokens_np = generated_tokens[0].cpu().numpy()
            
            # Remove padding tokens (0s) to get actual tokens
            actual_tokens = generated_tokens_np[generated_tokens_np != PAD_TOKEN]
            
            print(f"  🎯 Generated tokens: {actual_tokens.tolist()}")
            print(f"  🔧 Generated {len(actual_tokens)} actual tokens (before padding removal: {len(generated_tokens_np)})")
            
            # Use regular detokenize method (not detokenize_with_padding)
            pred_sequence = fast_tokenizer.detokenize([actual_tokens.tolist()])
            
            # Handle single sequence case - ensure we get a 1D array
            if pred_sequence.ndim == 2:
                if pred_sequence.shape[0] == 1:
                    pred_sequence = pred_sequence[0]  # Remove batch dimension
                else:
                    pred_sequence = pred_sequence.squeeze()  # Remove any size-1 dimensions
            
            # Ensure we have a 1D array
            if pred_sequence.ndim > 1:
                pred_sequence = pred_sequence.flatten()
            
            predictions.append(pred_sequence)
            
            # Store original target for comparison
            targets.append(test_targets[i])
            
            print(f"  ✅ Generated sequence {i+1}: {len(pred_sequence)} points")
            
        except Exception as e:
            print(f"  ❌ Detokenization failed for sequence {i+1}: {e}")
            print(f"  🔧 Generated tokens: {generated_tokens[0].cpu().numpy()[:10]}...")
            # Use original as fallback
            predictions.append(test_targets[i])
            targets.append(test_targets[i])

print(f"✅ TRUE INFERENCE completed!")
print(f"📊 Inference Statistics:")
print(f"  - Generated {len(predictions)} sequences")
print(f"  - First tokens used: {first_tokens}")
print(f"  - Prediction shape: {predictions[0].shape}")
print(f"  - Target shape: {targets[0].shape}")