from fast_tokenizer import FASTTokenizer
from cubic_spline_generator import CubicSplineGenerator
from transformer_model import SimpleTransformer
from training import FASTTrainer

import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PARAMETERS
# =============================================================================
NUM_SEQUENCES = 10000
SEQUENCE_LENGTH = 800
SEED = 42
MAX_TOKEN_LENGTH = 40
PAD_TOKEN = 0
VOCAB_SIZE = 2048
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
MAX_SEQ_LEN = 50
CONDITIONING_DIM = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
PATIENCE = 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"📊 Configuration:")
print(f"  - Dataset: {NUM_SEQUENCES:,} sequences, length {SEQUENCE_LENGTH}")
print(f"  - Tokenization: max_length={MAX_TOKEN_LENGTH}, vocab_size={VOCAB_SIZE}")
print(f"  - Model: d_model={D_MODEL}, nhead={NHEAD}, layers={NUM_LAYERS}")
print(f"  - Training: epochs={NUM_EPOCHS}, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}")
print(f"  - Device: {DEVICE}")

# =============================================================================
# GENERATE DATA
# =============================================================================
print(f"\n🔄 Generating dataset...")
generator = CubicSplineGenerator(seed=SEED)
times, targets, conditioning = generator.generate_spline_data(
    num_sequences=NUM_SEQUENCES,
    sequence_length=SEQUENCE_LENGTH
)
print(f"✅ Dataset generated: {targets.shape}")

# =============================================================================
# INITIALIZE TOKENIZER AND MODEL
# =============================================================================
print(f"\n🔧 Initializing tokenizer and model...")
fast_tokenizer = FASTTokenizer()

model = SimpleTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    conditioning_dim=CONDITIONING_DIM
).to(DEVICE)

trainer = FASTTrainer(
    model=model,
    tokenizer=fast_tokenizer,
    device=DEVICE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    vocab_size=VOCAB_SIZE,
    pad_token=PAD_TOKEN
)
print(f"✅ Initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

# =============================================================================
# TRAINING
# =============================================================================
print(f"\n📊 Preparing data...")
train_loader, val_loader = trainer.prepare_data(
    times=times,
    targets=targets,
    conditioning=conditioning,
    train_ratio=0.8
)

# print(f"🚀 Starting training...")
# train_losses = []
# val_losses = []
# best_val_loss = float('inf')
# patience_counter = 0

# for epoch in range(NUM_EPOCHS):
#     train_loss = trainer.train_epoch(train_loader)
#     val_loss = trainer.validate(val_loader)
    
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
    
#     if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
#         print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         patience_counter = 0
#         torch.save(model.state_dict(), 'best_fast_model.pth')
#     else:
#         patience_counter += 1
#         if patience_counter >= PATIENCE:
#             print(f"🛑 Early stopping at epoch {epoch}")
#             break

# print(f"✅ Training completed! Best val loss: {best_val_loss:.4f}")

# # =============================================================================
# # GENERATE AND VISUALIZE
# # =============================================================================
# model.load_state_dict(torch.load('best_fast_model.pth'))
# model.eval()

print(f"\n🎯 Generating test sequences...")
test_generator = CubicSplineGenerator(seed=999)
test_times, test_targets, test_conditioning = test_generator.generate_spline_data(
    num_sequences=4,
    sequence_length=SEQUENCE_LENGTH
)

test_tokens = fast_tokenizer.tokenize_with_padding(
    test_targets,
    pad_token=PAD_TOKEN,
    max_length=MAX_TOKEN_LENGTH,
    vocab_size=VOCAB_SIZE
)


predictions = []
with torch.no_grad():
    for i in range(len(test_targets)):
        first_token = torch.from_numpy(test_tokens[i:i+1, 0:1]).long().to(DEVICE)
        conditioning_tensor = torch.from_numpy(test_conditioning[i:i+1]).float().to(DEVICE)
        
        generated_tokens = model.generate(
            conditioning_points=conditioning_tensor,
            start_tokens=first_token,
            max_length=MAX_TOKEN_LENGTH,
            greedy=True,
            device=DEVICE,
            tokenizer=fast_tokenizer
        )
        
        actual_tokens = generated_tokens[0].cpu().numpy()
        actual_tokens = actual_tokens[actual_tokens != PAD_TOKEN]
        
        pred_sequence = fast_tokenizer.detokenize([test_tokens.tolist()])
        if pred_sequence.ndim > 1:
            pred_sequence = pred_sequence.flatten()
        
        # Match length
        if len(pred_sequence) != len(test_targets[i]):
            if len(pred_sequence) < len(test_targets[i]):
                pred_sequence = np.pad(pred_sequence, (0, len(test_targets[i]) - len(pred_sequence)),
                                     mode='constant', constant_values=pred_sequence[-1] if len(pred_sequence) > 0 else 0)
            else:
                pred_sequence = pred_sequence[:len(test_targets[i])]
        
        predictions.append(pred_sequence)

# =============================================================================
# VISUALIZATION
# =============================================================================
print(f"\n🎨 Visualizing generated sequences...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i in range(4):
    ax = axes[i]
    ax.plot(test_targets[i], label='Target', alpha=0.8, linewidth=2, color='blue')
    ax.plot(predictions[i], label='Generated', alpha=0.8, linewidth=2, linestyle='--', color='red')
    mse = np.mean((test_targets[i] - predictions[i])**2)
    ax.set_title(f'Sequence {i+1} (MSE: {mse:.4f})')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"✅ Done!")
