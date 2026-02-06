"""
Merge all datasets (existing + synthetic) for final training
Quick script to combine LineVul, Production, and Synthetic data
"""

import json
from pathlib import Path
import random

print("="*60)
print("MERGING ALL DATASETS")
print("="*60)

all_samples = []

# Load existing processed data (from LineVul + Production)
existing_files = [
    'backend/ml/data/train_dataset.json',
    'backend/ml/data/val_dataset.json',
    'backend/ml/data/test_dataset.json'
]

for filepath in existing_files:
    path = Path(filepath)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            all_samples.extend(data['samples'])
            print(f"✓ Loaded {len(data['samples'])} from {path.name}")

# Load synthetic data
synthetic_files = [
    'backend/ml/data/train_dataset.json',  # From generate_synthetic
    'backend/ml/data/val_dataset.json',
    'backend/ml/data/test_dataset.json'
]

# Actually, generate_synthetic created these files, so they're already loaded
# Let's check what we have

print(f"\nTotal samples before deduplication: {len(all_samples)}")

# Remove duplicates based on code hash
unique_samples = []
seen_hashes = set()

for sample in all_samples:
    code_hash = hash(sample['code'][:200] if len(sample['code']) > 200 else sample['code'])
    if code_hash not in seen_hashes:
        seen_hashes.add(code_hash)
        unique_samples.append(sample)

print(f"Total samples after deduplication: {len(unique_samples)}")

# Shuffle
random.shuffle(unique_samples)

# Split 70/15/15
n = len(unique_samples)
train_size = int(n * 0.7)
val_size = int(n * 0.15)

train_data = unique_samples[:train_size]
val_data = unique_samples[train_size:train_size+val_size]
test_data = unique_samples[train_size+val_size:]

print(f"\n📊 Final Split:")
print(f"  Train: {len(train_data)}")
print(f"  Val:   {len(val_data)}")
print(f"  Test:  {len(test_data)}")

# Calculate class weights
from collections import Counter
labels = [int(s['is_vulnerable']) for s in unique_samples]
counter = Counter(labels)

class_weights = {
    "0": 1.0 / counter[0] * len(labels) / 2 if 0 in counter else 1.0,
    "1": 1.0 / counter[1] * len(labels) / 2 if 1 in counter else 1.0
}

print(f"\n⚖️  Class Balance:")
print(f"  Safe (0): {counter.get(0, 0)} samples (weight: {class_weights['0']:.4f})")
print(f"  Vuln (1): {counter.get(1, 0)} samples (weight: {class_weights['1']:.4f})")

# Save final datasets
output_dir = Path('backend/ml/data/final')
output_dir.mkdir(exist_ok=True)

for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    output_file = output_dir / f"{name}_dataset.json"
    with open(output_file, 'w') as f:
        json.dump({
            'samples': data,
            'class_weights': class_weights,
            'metadata': {
                'total_samples': len(data),
                'source': 'merged',
                'includes': ['linevul', 'production', 'synthetic']
            }
        }, f, indent=2)
    print(f"✓ Saved {output_file}")

print("\n✅ Dataset merge complete!")
print(f"📁 Final dataset location: backend/ml/data/final/")
print(f"\n🚀 Ready to train!")
print(f"Run: python backend/ml/train.py")
