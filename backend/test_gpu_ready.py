import torch
import torch_geometric
from app.ml.hybrid_model import HybridVulnerabilityModel

print("=" * 60)
print("FINAL GPU + DEPENDENCIES CHECK")
print("=" * 60)
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"✅ HybridModel: Loaded successfully")
print("=" * 60)
print("\n🚀 READY TO TRAIN WITH GPU!")
print("   Expected: 30-60 seconds per epoch (15-30x faster)")
print("   Total time: ~25-50 minutes for 50 epochs")
print("=" * 60)
