import pickle
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, 'scripts')
from enhanced_dataset_pipeline import ProcessedSample

# Load processed samples
with open('data/processed_graphs/train_graphs.pkl', 'rb') as f:
    samples = pickle.load(f)

print(f"✅ Loaded {len(samples)} training samples\n")

# Check first sample
s = samples[0]
print("Sample 0:")
print(f"  Code: {s.code[:60]}...")
print(f"  Language: {s.language}")
print(f"  Label: {s.label} ({'vulnerable' if s.label == 1 else 'safe'})")
print(f"  Vuln type: {s.vulnerability_type}")
print(f"\n  Token IDs in sample: {s.token_ids.shape if s.token_ids is not None else 'None'}")
print(f"  Token IDs in graph_data: {s.graph_data.token_ids.shape if hasattr(s.graph_data, 'token_ids') else 'Not attached'}")
print(f"  Graph nodes: {s.graph_data.x.shape[0]}")
print(f"  Graph edges: {s.graph_data.edge_index.shape[1]}")

# Check vocabulary
vocab_path = Path('data/processed_graphs/vocabulary.pkl')
if vocab_path.exists():
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    print(f"\n✅ Vocabulary: {len(vocab_data['vocab'])} tokens")
    print(f"  Max sequence length: {vocab_data['max_seq_length']}")
else:
    print("\n⚠️ No vocabulary file found")

print("\n✅ Token extraction successful!")
