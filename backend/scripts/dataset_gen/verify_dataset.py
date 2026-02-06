"""
Simple dataset verification - check what we have and if it's ready to train
"""
import json
from pathlib import Path

print("="*60)
print("DATASET VERIFICATION")
print("="*60)

data_dir = Path('backend/ml/data')

files = ['train_dataset.json', 'val_dataset.json', 'test_dataset.json']

total_samples = 0

for filename in files:
    filepath = data_dir / filename
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
            n_samples = len(data['samples'])
            total_samples += n_samples
            
            # Check first sample
            if n_samples > 0:
                sample = data['samples'][0]
                print(f"\n📄 {filename}:")
                print(f"  Samples: {n_samples}")
                print(f"  Keys: {list(sample.keys())}")
                
                # Check if has vulnerabilities
                if 'vulnerabilities' in sample:
                    print(f"  Format: Multi-vuln (has 'vulnerabilities' list)")
                    print(f"  First vuln: {sample['vulnerabilities'][0] if sample['vulnerabilities'] else 'None'}")
                else:
                    print(f"  Format: Unknown/Legacy")

print(f"\n📊 TOTAL: {total_samples} samples")

if total_samples >= 3000:
    print(f"\n✅ Dataset is ready for training!")
    print(f"   {total_samples} samples is sufficient for GNN+LSTM")
    print(f"\n🚀 Next step: python backend/ml/train.py")
else:
    print(f"\n⚠️  Dataset might be small ({total_samples} samples)")
    print(f"   Recommended: 3,000+ samples")
    print(f"   But can still train - accuracy may be lower")
