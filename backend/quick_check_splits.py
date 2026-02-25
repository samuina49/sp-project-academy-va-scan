import pickle
from pathlib import Path
from collections import Counter
import sys

# Import ProcessedSample for pickle deserialization
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

# Register module for pickle
import importlib.util
spec = importlib.util.spec_from_file_location(
    "enhanced_dataset_pipeline",
    Path(__file__).parent / "scripts" / "enhanced_dataset_pipeline.py"
)
edp_module = importlib.util.module_from_spec(spec)
sys.modules['scripts.enhanced_dataset_pipeline'] = edp_module
spec.loader.exec_module(edp_module)

def quick_check_split(split_name):
    """Quick check of a dataset split"""
    path = Path(f"data/processed_graphs/{split_name}_graphs.pkl")
    
    print(f"\n{'='*60}")
    print(f"Checking {split_name.upper()} SET")
    print(f"{'='*60}")
    
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    
    print(f"Total samples: {len(samples)}")
    
    # Language distribution
    languages = Counter(s.language for s in samples)
    print(f"\nLanguages:")
    for lang, count in languages.items():
        pct = count / len(samples) * 100
        print(f"  • {lang}: {count} ({pct:.1f}%)")
    
    # Label distribution
    labels = Counter(s.label for s in samples)
    print(f"\nLabels:")
    vuln = labels.get(1, 0)
    safe = labels.get(0, 0)
    print(f"  • Vulnerable: {vuln} ({vuln/len(samples)*100:.1f}%)")
    print(f"  • Safe: {safe} ({safe/len(samples)*100:.1f}%)")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        quick_check_split(split)
    
    print(f"\n{'='*60}")
    print("✅ All splits verified!")
    print(f"{'='*60}")
