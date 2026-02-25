"""Quick test to verify model loads correctly with new paths"""
import sys
sys.path.insert(0, '.')

try:
    from app.api.v1.ai_scan import load_model
    print("Attempting to load model...")
    model, extractor, scanner = load_model()
    print("✅ SUCCESS: Model loaded without errors")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Model vocab size: {model.lstm_branch.embedding.weight.shape[0]}")
    print(f"   Node feature dim: {model.gnn_branch.conv_layers[0].lin.weight.shape[1]}")
    print(f"   Has metrics branch: {hasattr(model, 'metrics_branch')}")
    print(f"   Device: {next(model.parameters()).device}")
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
