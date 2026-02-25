import torch
from pathlib import Path

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Check old model
old_model = Path("training/models/hybrid_model_best.pth")
if old_model.exists():
    print("\n1. OLD MODEL (Currently Used by Scanner):")
    print(f"   Path: {old_model}")
    print(f"   Size: {old_model.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Modified: {old_model.stat().st_mtime}")
    try:
        old_checkpoint = torch.load(old_model, map_location='cpu', weights_only=False)
        if isinstance(old_checkpoint, dict):
            print(f"   Accuracy: 90.86% (from config comments)")
    except Exception as e:
        print(f"   Error: {e}")
else:
    print("\n1. OLD MODEL: Not found")

# Check new model
new_model = Path("training/checkpoints/best_model.pt")
if new_model.exists():
    print("\n2. NEW MODEL (Latest Training):")
    print(f"   Path: {new_model}")
    print(f"   Size: {new_model.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Modified: {new_model.stat().st_mtime}")
    try:
        checkpoint = torch.load(new_model, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   Val F1: {checkpoint.get('val_f1', 0)*100:.2f}%")
            print(f"   Val Acc: {checkpoint.get('val_acc', 0)*100:.2f}%")
            print(f"   Val Loss: {checkpoint.get('val_loss', 0):.4f}")
            print(f"   Train Loss: {checkpoint.get('train_loss', 0):.4f}")
            
            # Check if it has the model state
            if 'model_state_dict' in checkpoint:
                print("   Status: Complete checkpoint (has model_state_dict)")
            else:
                print("   Status: Incomplete checkpoint")
        else:
            print("   Status: Direct model (not checkpoint dict)")
    except Exception as e:
        print(f"   Error: {e}")
else:
    print("\n2. NEW MODEL: Not found")

print("\n" + "=" * 60)
print("CONCLUSION:")
if new_model.exists():
    print("✅ New model exists but scanner is NOT using it yet")
    print("   Scanner uses: training/models/hybrid_model_best.pth")
    print("   Latest model: training/checkpoints/best_model.pt")
    print("\nTO USE NEW MODEL:")
    print("   Update ML_MODEL_PATH in app/core/config.py")
else:
    print("⚠️  New model not found - training may have failed")
print("=" * 60)
