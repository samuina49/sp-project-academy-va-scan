"""
Enhanced Dataset Pipeline - Process Real Code to Graphs
========================================================
Uses the Enhanced Graph Builder to convert real-world code
into graph representations (AST + CFG + DFG) for training.

Features:
- Load datasets from multiple sources
- Extract graphs using tree-sitter + Enhanced Graph Builder
- Save PyTorch Geometric Data format
- Train/Val/Test split
- Statistics and quality checks

Author: AI Vulnerability Scanner
Date: February 6, 2026
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import logging
import pickle
from dataclasses import dataclass, asdict
import sys
import os
import re
from collections import Counter

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# Import our enhanced graph builder
from app.ml.enhanced_graph_builder import EnhancedFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """
    Simple tokenizer for code.
    Extracts tokens (identifiers, keywords, operators, literals) from source code.
    """
    
    def __init__(self, max_vocab_size: int = 10000, max_seq_length: int = 512):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        
        # Vocabulary
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.START_TOKEN: 2,
            self.END_TOKEN: 3
        }
        self.token_to_id = self.vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_built = False
        
        # Token patterns for code
        self.token_pattern = re.compile(
            r'\b\w+\b|'           # Identifiers and keywords
            r'[+\-*/%=<>!&|^~]+'  # Operators
            r'|[(){}\[\];:,.]'    # Punctuation
            r'|\"[^\"]*\"|\'[^\']*\''  # Strings
            r'|\d+\.?\d*'         # Numbers
        )
    
    def tokenize(self, code: str) -> List[str]:
        """Tokenize code into tokens"""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # JS/C++ comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Block comments
        
        # Extract tokens
        tokens = self.token_pattern.findall(code)
        return tokens
    
    def build_vocabulary(self, code_samples: List[Tuple[str, str]]):
        """
        Build vocabulary from code samples.
        
        Args:
            code_samples: List of (code, language) tuples
        """
        logger.info("🔤 Building vocabulary from training samples...")
        
        # Count token frequencies
        token_counter = Counter()
        
        for code, lang in tqdm(code_samples, desc="Tokenizing"):
            tokens = self.tokenize(code)
            token_counter.update(tokens)
        
        # Take top N most frequent tokens
        most_common = token_counter.most_common(self.max_vocab_size - len(self.vocab))
        
        # Add to vocabulary
        for token, count in most_common:
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
        
        self.vocab_built = True
        
        logger.info(f"  ✅ Vocabulary built: {len(self.vocab)} tokens")
        logger.info(f"  📊 Unique tokens in training: {len(token_counter)}")
        logger.info(f"  🔝 Most common: {token_counter.most_common(10)}")
    
    def encode(self, code: str) -> torch.Tensor:
        """
        Encode code to token IDs.
        
        Args:
            code: Source code string
            
        Returns:
            Tensor of token IDs [1, seq_len]
        """
        if not self.vocab_built:
            logger.warning("Vocabulary not built yet! Using default vocab.")
        
        # Tokenize
        tokens = self.tokenize(code)
        
        # Add START/END tokens
        tokens = [self.START_TOKEN] + tokens + [self.END_TOKEN]
        
        # Truncate if too long
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        # Convert to IDs
        token_ids = [
            self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
            for token in tokens
        ]
        
        # Pad if too short
        while len(token_ids) < self.max_seq_length:
            token_ids.append(self.token_to_id[self.PAD_TOKEN])
        
        # Convert to tensor [1, seq_len]
        return torch.tensor([token_ids], dtype=torch.long)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to code"""
        if token_ids.dim() == 2:
            token_ids = token_ids[0]  # Remove batch dimension
        
        tokens = [
            self.id_to_token.get(int(tid), self.UNK_TOKEN)
            for tid in token_ids
            if int(tid) != self.token_to_id[self.PAD_TOKEN]
        ]
        
        return ' '.join(tokens)
    
    def save(self, path: Path):
        """Save vocabulary to file"""
        vocab_data = {
            'vocab': self.vocab,
            'max_vocab_size': self.max_vocab_size,
            'max_seq_length': self.max_seq_length
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
        logger.info(f"Vocabulary saved to {path}")
    
    def load(self, path: Path):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.vocab = vocab_data['vocab']
        self.token_to_id = self.vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.max_vocab_size = vocab_data['max_vocab_size']
        self.max_seq_length = vocab_data['max_seq_length']
        self.vocab_built = True
        
        logger.info(f"Vocabulary loaded from {path}: {len(self.vocab)} tokens")


@dataclass
class ProcessedSample:
    """Processed code sample with graph representation"""
    code: str
    label: int  # 0 = safe, 1 = vulnerable
    language: str
    graph_data: any  # PyTorch Geometric Data
    vulnerability_type: str
    source: str
    metadata: Dict
    token_ids: Optional[torch.Tensor] = None  # Token sequence for LSTM [1, seq_len]


class EnhancedDatasetPipeline:
    """
    Process raw code datasets into graph representations.
    """
    
    def __init__(
        self,
        raw_data_dir: str = "data/raw_datasets",
        output_dir: str = "data/processed_graphs",
        max_samples: int = None,
        build_vocab: bool = True,
        max_seq_length: int = 512
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_samples = max_samples
        self.build_vocab = build_vocab
        
        # Initialize graph extractor
        self.graph_extractor = EnhancedFeatureExtractor(
            max_seq_length=512,
            node_feature_dim=64
        )
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(
            max_vocab_size=10000,
            max_seq_length=max_seq_length
        )
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'by_language': {},
            'by_label': {},
            'avg_nodes': 0,
            'avg_edges': 0,
            'avg_cfg_edges': 0,
            'avg_dfg_edges': 0
        }
        
        logger.info("✅ Enhanced Dataset Pipeline initialized")
    
    def process_all_datasets(self):
        """Process all available datasets"""
        logger.info("="*70)
        logger.info("🚀 ENHANCED DATASET PROCESSING PIPELINE")
        logger.info("="*70)
        
        # Find all JSON datasets (non-recursive to avoid subfolders)
        dataset_files = []
        if self.raw_data_dir.exists():
            dataset_files = list(self.raw_data_dir.glob("*.json"))
            dataset_files = [f for f in dataset_files if f.stat().st_size > 0]
        
        logger.info(f"\n📦 Found {len(dataset_files)} dataset files")
        
        if not dataset_files:
            logger.warning("⚠️ No dataset files found!")
            logger.info(f"📁 Looking in: {self.raw_data_dir.absolute()}")
            logger.info("\n💡 Please download datasets first:")
            logger.info("   python scripts/download_quality_datasets.py")
            logger.info("   OR")
            logger.info("   python scripts/quick_download_datasets.py")
            return
        
        # Load all samples first (for vocabulary building)
        logger.info("\n📖 Loading all datasets...")
        all_samples = []
        for dataset_file in dataset_files:
            try:
                samples = self._load_dataset(dataset_file)
                all_samples.extend([(s, dataset_file.stem) for s in samples])
            except Exception as e:
                logger.error(f"Failed to load {dataset_file.name}: {e}")
        
        logger.info(f"  ✅ Loaded {len(all_samples)} total samples")
        
        # Build vocabulary from all training samples (if enabled)
        if self.build_vocab and all_samples:
            logger.info("\n🔤 Building vocabulary from all samples...")
            code_samples = [
                (s['code'], s.get('language', 'python'))
                for s, _ in all_samples
                if 'code' in s and s['code']
            ]
            self.tokenizer.build_vocabulary(code_samples)
            
            # Save vocabulary
            vocab_path = self.output_dir / 'vocabulary.pkl'
            self.tokenizer.save(vocab_path)
        
        # Process each dataset
        all_processed_samples = []
        
        for dataset_file in dataset_files:
            logger.info(f"\n📄 Processing: {dataset_file.name}")
            
            try:
                samples = self._load_dataset(dataset_file)
                processed = self._process_dataset(samples, dataset_file.stem)
                all_processed_samples.extend(processed)
                
                logger.info(f"  ✅ Processed {len(processed)} samples from {dataset_file.name}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to process {dataset_file.name}: {e}")
                continue
        
        # Save processed datasets
        if all_processed_samples:
            self._save_processed_datasets(all_processed_samples)
            self._print_statistics()
        else:
            logger.warning("⚠️ No samples were successfully processed!")
    
    def _load_dataset(self, file_path: Path) -> List[Dict]:
        """Load dataset from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys
            for key in ['data', 'samples', 'examples']:
                if key in data:
                    return data[key]
            # If no common key, return as single sample
            return [data]
        
        return []
    
    def _process_dataset(
        self,
        samples: List[Dict],
        dataset_name: str
    ) -> List[ProcessedSample]:
        """Process dataset samples into graphs"""
        processed_samples = []
        
        # Limit samples if max_samples is set
        if self.max_samples:
            samples = samples[:self.max_samples]
        
        progress_bar = tqdm(samples, desc=f"Processing {dataset_name}")
        
        for sample in progress_bar:
            try:
                # Extract fields
                code = sample.get('code', '')
                label = int(sample.get('label', 0))
                language = sample.get('language', 'python').lower()
                vuln_type = sample.get('vulnerability_type', 'unknown')
                source = sample.get('source', dataset_name)
                
                # Validate code
                if not code or len(code) < 20:
                    continue
                
                # Skip unsupported languages
                if language not in ['python', 'javascript', 'typescript']:
                    continue
                
                # Extract graph using Enhanced Graph Builder
                graph_data = self.graph_extractor.extract_enhanced_graph(
                    code, language
                )
                
                # Validate graph
                if graph_data.x.shape[0] < 2:  # Too small
                    continue
                
                # Tokenize code for LSTM branch
                token_ids = None
                if self.tokenizer.vocab_built:
                    try:
                        token_ids = self.tokenizer.encode(code)  # [1, seq_len]
                        # Attach to graph_data for batching
                        graph_data.token_ids = token_ids
                    except Exception as e:
                        logger.debug(f"Failed to tokenize: {e}")
                
                # Create processed sample
                processed = ProcessedSample(
                    code=code,
                    label=label,
                    language=language,
                    graph_data=graph_data,
                    vulnerability_type=vuln_type,
                    source=source,
                    metadata=sample.get('metadata', {}),
                    token_ids=token_ids
                )
                
                processed_samples.append(processed)
                
                # Update statistics
                self._update_stats(processed, graph_data)
                
            except Exception as e:
                self.stats['failed'] += 1
                logger.debug(f"Failed to process sample: {e}")
                continue
        
        return processed_samples
    
    def _update_stats(self, sample: ProcessedSample, graph_data):
        """Update processing statistics"""
        self.stats['total_processed'] += 1
        self.stats['successful'] += 1
        
        # Language stats
        lang = sample.language
        if lang not in self.stats['by_language']:
            self.stats['by_language'][lang] = 0
        self.stats['by_language'][lang] += 1
        
        # Label stats
        label = 'vulnerable' if sample.label == 1 else 'safe'
        if label not in self.stats['by_label']:
            self.stats['by_label'][label] = 0
        self.stats['by_label'][label] += 1
        
        # Graph stats
        num_nodes = graph_data.x.shape[0]
        num_edges = graph_data.edge_index.shape[1]
        self.stats['avg_nodes'] += num_nodes
        self.stats['avg_edges'] += num_edges
        
        # Count edge types
        if graph_data.edge_attr.shape[0] > 0:
            cfg_edges = (graph_data.edge_attr == 2).sum().item()
            dfg_edges = (graph_data.edge_attr == 1).sum().item()
            self.stats['avg_cfg_edges'] += cfg_edges
            self.stats['avg_dfg_edges'] += dfg_edges
    
    def _save_processed_datasets(self, samples: List[ProcessedSample]):
        """Save processed samples"""
        logger.info(f"\n💾 Saving {len(samples)} processed samples...")
        
        # Split into train/val/test
        train_size = int(0.7 * len(samples))
        val_size = int(0.15 * len(samples))
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        for split_name, split_samples in splits.items():
            # Save as pickle (preserves torch tensors)
            output_file = self.output_dir / f"{split_name}_graphs.pkl"
            
            with open(output_file, 'wb') as f:
                pickle.dump(split_samples, f)
            
            logger.info(f"  ✅ Saved {len(split_samples)} samples to {output_file.name}")
        
        # Save metadata
        metadata = {
            'total_samples': len(samples),
            'train_size': len(train_samples),
            'val_size': len(val_samples),
            'test_size': len(test_samples),
            'statistics': self.stats
        }
        
        meta_file = self.output_dir / "dataset_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  ✅ Saved metadata to {meta_file.name}")
    
    def _print_statistics(self):
        """Print processing statistics"""
        if self.stats['successful'] == 0:
            return
        
        logger.info("\n" + "="*70)
        logger.info("📊 PROCESSING STATISTICS")
        logger.info("="*70)
        
        logger.info(f"\n✅ Successfully processed: {self.stats['successful']:,}")
        logger.info(f"❌ Failed: {self.stats['failed']:,}")
        
        # Language distribution
        logger.info(f"\n📝 Language Distribution:")
        for lang, count in self.stats['by_language'].items():
            percentage = (count / self.stats['successful']) * 100
            logger.info(f"  • {lang}: {count:,} ({percentage:.1f}%)")
        
        # Label distribution
        logger.info(f"\n🏷️  Label Distribution:")
        for label, count in self.stats['by_label'].items():
            percentage = (count / self.stats['successful']) * 100
            logger.info(f"  • {label}: {count:,} ({percentage:.1f}%)")
        
        # Average graph stats
        n = self.stats['successful']
        logger.info(f"\n📈 Average Graph Statistics:")
        logger.info(f"  • Nodes per graph: {self.stats['avg_nodes'] / n:.1f}")
        logger.info(f"  • Edges per graph: {self.stats['avg_edges'] / n:.1f}")
        logger.info(f"  • CFG edges per graph: {self.stats['avg_cfg_edges'] / n:.1f}")
        logger.info(f"  • DFG edges per graph: {self.stats['avg_dfg_edges'] / n:.1f}")
        
        logger.info("\n" + "="*70)
        logger.info(f"📁 Output directory: {self.output_dir.absolute()}")
        logger.info("="*70)
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Review processed graphs: ls data/processed_graphs/")
        logger.info("2. Train model: python training/train_with_enhanced_graphs.py")
        logger.info("3. Evaluate results")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process vulnerability datasets into graphs")
    parser.add_argument("--data-dir", type=str, default="data/raw_datasets",
                       help="Directory containing raw JSON datasets")
    parser.add_argument("--output-dir", type=str, default="data/processed_graphs",
                       help="Directory to save processed graphs")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Max samples per dataset for testing")
    args = parser.parse_args()
    
    # Check if datasets exist
    raw_data_dir = Path(args.data_dir)
    if not raw_data_dir.exists() or not list(raw_data_dir.glob("*.json")):
        print("\n" + "="*70)
        print("⚠️ WARNING: No datasets found!")
        print("="*70)
        print(f"\n📁 Looking in: {raw_data_dir.absolute()}")
        print("\n📦 Please download datasets first:")
        print("\n Option 1: Quick download (Hugging Face)")
        print("   python scripts/quick_download_datasets.py")
        print("\n Option 2: Manual download")
        print("   See DOWNLOAD_DATASETS_GUIDE.md for instructions")
        print("\n" + "="*70)
        return
    
    # Initialize pipeline
    pipeline = EnhancedDatasetPipeline(
        raw_data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )
    
    # Process all datasets
    try:
        pipeline.process_all_datasets()
        print("\n✅ Dataset processing complete!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
