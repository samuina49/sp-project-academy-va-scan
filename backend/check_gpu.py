import torch

print("=" * 60)
print("GPU CHECK - RTX 4060")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("\n✅ GPU READY TO USE!")
    print(f"Expected speedup: 10-50x faster than CPU")
else:
    print(f"CUDA version: Not available")
    print("\n❌ GPU NOT DETECTED")
    print("Need to install PyTorch with CUDA support:")
    print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("=" * 60)
