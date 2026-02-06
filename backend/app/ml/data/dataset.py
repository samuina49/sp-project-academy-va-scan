"""
PyTorch Dataset for Vulnerability Detection

This module provides a Dataset class for loading and processing
code samples for training the GNN + LSTM model.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

from ..data.ast_parser import parse_code_to_ast
from ..data.graph_builder import GraphBuilder, GraphData
from ..data.tokenizer import CodeTokenizer


class VulnerabilityDataset(Dataset):
    """
    Dataset for code vulnerability detection
    
    Each sample contains:
    - Source code
    - Label (0: safe, 1: vulnerable)
    - Metadata (vulnerability type, severity, etc.)
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer: CodeTokenizer,
        graph_builder: GraphBuilder,
        max_graph_nodes: int = 1000,
        max_seq_length: int = 512,
        language: str = "python"
    ):
        """
        Initialize dataset
        
        Args:
            data_file: Path to JSON file with samples
            tokenizer: Code tokenizer instance
            graph_builder: Graph builder instance
            max_graph_nodes: Maximum nodes in graph
            max_seq_length: Maximum sequence length
            language: Programming language
        """
        self.tokenizer = tokenizer
        self.graph_builder = graph_builder
        self.max_graph_nodes = max_graph_nodes
        self.max_seq_length = max_seq_length
        self.language = language
        
        # Load data
        self.samples = self._load_data(data_file)
        
    def _load_data(self, data_file: str) -> List[Dict]:
        """Load samples from JSON file"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[GraphData, torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (graph_data, token_ids, attention_mask, label)
        """
        sample = self.samples[idx]
        code = sample['code']
        label = sample['label']  # 0 or 1
        
        # Process for GNN
        ast_nodes = parse_code_to_ast(code, self.language)
        graph_data = self.graph_builder.build_graph(ast_nodes)
        
        # Process for LSTM
        token_ids = self.tokenizer.encode(code, add_special_tokens=True)
        
        # Truncate if needed
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad to max length
        padding_length = self.max_seq_length - len(token_ids)
        token_ids = token_ids + [self.tokenizer.token_to_id[self.tokenizer.PAD_TOKEN]] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        # Convert to tensors
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return graph_data, token_ids, attention_mask, label


def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for batching graphs
    
    Graphs have different sizes, so we need special handling.
    
    Args:
        batch: List of (graph_data, token_ids, attention_mask, label)
        
    Returns:
        Tuple of batched (graphs, token_ids, attention_masks, labels)
    """
    graphs, token_ids_list, attention_masks, labels = zip(*batch)
    
    # Stack token IDs and attention masks (same size already)
    token_ids = torch.stack(token_ids_list)
    attention_mask = torch.stack(attention_masks)
    labels = torch.stack(labels)
    
    # Batch graphs
    # Concatenate all node features
    all_node_features = []
    all_edge_indices = []
    batch_assignment = []
    
    node_offset = 0
    for batch_idx, graph in enumerate(graphs):
        all_node_features.append(graph.node_features)
        
        # Adjust edge indices by node offset
        adjusted_edges = graph.edge_index + node_offset
        all_edge_indices.append(adjusted_edges)
        
        # Batch assignment
        batch_assignment.extend([batch_idx] * graph.num_nodes)
        
        node_offset += graph.num_nodes
    
    # Concatenate
    batched_node_features = torch.cat(all_node_features, dim=0)
    batched_edge_index = torch.cat(all_edge_indices, dim=1)
    batch_tensor = torch.tensor(batch_assignment, dtype=torch.long)
    
    return (batched_node_features, batched_edge_index, batch_tensor), token_ids, attention_mask, labels


# Example usage
if __name__ == "__main__":
    # Create sample data file
    sample_data = [
        {
            "code": "import os\ndef unsafe(x):\n    os.system(x)",
            "label": 1,
            "metadata": {"type": "command_injection", "severity": "high"}
        },
        {
            "code": "def safe(x):\n    return x + 1",
            "label": 0,
            "metadata": {}
        }
    ]
    
    # Save sample data
    with open("sample_data.json", "w") as f:
        json.dump(sample_data, f)
    
    # Create tokenizer and graph builder
    tokenizer = CodeTokenizer(vocab_size=1000, max_length=128)
    tokenizer.train([s['code'] for s in sample_data])
    
    graph_builder = GraphBuilder(node_feature_dim=64)
    
    # Create dataset
    dataset = VulnerabilityDataset(
        data_file="sample_data.json",
        tokenizer=tokenizer,
        graph_builder=graph_builder
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    graph_data, token_ids, attention_mask, label = dataset[0]
    print(f"\nSample 0:")
    print(f"  Graph nodes: {graph_data.num_nodes}")
    print(f"  Graph edges: {graph_data.num_edges}")
    print(f"  Token IDs shape: {token_ids.shape}")
    print(f"  Label: {label}")
    
    # Test collate
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    for batch in loader:
        (node_feat, edge_idx, batch_tensor), token_ids, att_mask, labels = batch
        print(f"\nBatch:")
        print(f"  Node features: {node_feat.shape}")
        print(f"  Edge index: {edge_idx.shape}")
        print(f"  Batch: {batch_tensor.shape}")
        print(f"  Token IDs: {token_ids.shape}")
        print(f"  Labels: {labels.shape}")
        break
    
    # Clean up
    os.remove("sample_data.json")
