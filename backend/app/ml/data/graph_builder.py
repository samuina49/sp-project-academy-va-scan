"""
Graph Builder - Convert AST to Graph

This module converts Abstract Syntax Trees into graph representations
suitable for Graph Neural Networks (GNN).

The graph includes:
- AST edges (parent-child relationships)
- Control flow edges (program execution order)
- Data flow edges (variable dependencies)
"""

import torch
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .ast_parser import ASTNode


@dataclass
class GraphData:
    """
    Graph representation for GNN
    
    Attributes:
        node_features: Tensor of shape [num_nodes, feature_dim]
        edge_index: Tensor of shape [2, num_edges] (source, target pairs)
        edge_type: Tensor of shape [num_edges] (edge type indices)
        num_nodes: Number of nodes in graph
        num_edges: Number of edges in graph
    """
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    num_nodes: int
    num_edges: int
    node_mapping: Dict[int, int]  # Original ID to graph ID


class EdgeType:
    """Edge types in the code graph"""
    AST = 0          # Parent-child in AST
    CONTROL_FLOW = 1  # Program execution order
    DATA_FLOW = 2     # Variable dependencies
    NEXT_TOKEN = 3    # Sequential token order


class GraphBuilder:
    """
    Build graph representation from AST nodes
    """
    
    def __init__(self, node_feature_dim: int = 64):
        """
        Initialize graph builder
        
        Args:
            node_feature_dim: Dimension of node feature vectors
        """
        self.node_feature_dim = node_feature_dim
        self.node_type_vocab = self._build_node_type_vocab()
        
    def _build_node_type_vocab(self) -> Dict[str, int]:
        """
        Build vocabulary of AST node types
        
        Returns:
            Dictionary mapping node types to indices
        """
        # Common Python AST node types
        # In production, build this from training data
        common_types = [
            'Module', 'FunctionDef', 'ClassDef', 'Return', 'Delete',
            'Assign', 'AugAssign', 'For', 'While', 'If', 'With',
            'Raise', 'Try', 'Assert', 'Import', 'ImportFrom',
            'Expr', 'Pass', 'Break', 'Continue', 'Call', 'Name',
            'Constant', 'Attribute', 'Subscript', 'List', 'Tuple',
            'Dict', 'Set', 'Compare', 'BinOp', 'UnaryOp', 'Lambda',
            'IfExp', 'ListComp', 'SetComp', 'DictComp', 'GeneratorExp',
            'Await', 'Yield', 'YieldFrom', '<UNK>'  # Unknown type
        ]
        
        return {node_type: idx for idx, node_type in enumerate(common_types)}
    
    def build_graph(self, ast_nodes: List[ASTNode]) -> GraphData:
        """
        Convert AST nodes to graph representation
        
        Args:
            ast_nodes: List of AST nodes from parser
            
        Returns:
            GraphData object containing graph structure
        """
        if not ast_nodes:
            return self._empty_graph()
        
        # Build node mapping (original ID to sequential ID)
        node_mapping = {node.node_id: idx for idx, node in enumerate(ast_nodes)}
        num_nodes = len(ast_nodes)
        
        # Build node features
        node_features = self._build_node_features(ast_nodes)
        
        # Build edges
        edges, edge_types = self._build_edges(ast_nodes, node_mapping)
        
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=num_nodes,
            num_edges=len(edges),
            node_mapping=node_mapping
        )
    
    def _build_node_features(self, ast_nodes: List[ASTNode]) -> torch.Tensor:
        """
        Build feature vectors for each node
        
        Args:
            ast_nodes: List of AST nodes
            
        Returns:
            Tensor of shape [num_nodes, feature_dim]
        """
        features = []
        
        for node in ast_nodes:
            # One-hot encode node type
            node_type_idx = self.node_type_vocab.get(
                node.node_type, 
                self.node_type_vocab['<UNK>']
            )
            
            # Create feature vector
            # In a real implementation, this would be more sophisticated
            feature = torch.zeros(self.node_feature_dim)
            
            # Encode node type (first 40 dimensions for one-hot)
            if node_type_idx < self.node_feature_dim:
                feature[node_type_idx] = 1.0
            
            # Additional features (last dimensions)
            feature[-10] = float(len(node.children))  # Number of children
            feature[-9] = float(node.start_line)  # Line number
            feature[-8] = float(node.end_line - node.start_line)  # Line span
            feature[-7] = float(node.value is not None)  # Has value
            feature[-6] = float(len(node.attributes))  # Number of attributes
            
            features.append(feature)
        
        return torch.stack(features)
    
    def _build_edges(
        self, 
        ast_nodes: List[ASTNode], 
        node_mapping: Dict[int, int]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Build edges between nodes
        
        Args:
            ast_nodes: List of AST nodes
            node_mapping: Mapping from original ID to graph ID
            
        Returns:
            Tuple of (edge_list, edge_types)
        """
        edges = []
        edge_types = []
        
        # 1. AST edges (parent-child)
        for node in ast_nodes:
            parent_idx = node_mapping[node.node_id]
            
            for child_id in node.children:
                if child_id in node_mapping:
                    child_idx = node_mapping[child_id]
                    edges.append((parent_idx, child_idx))
                    edge_types.append(EdgeType.AST)
                    
                    # Add reverse edge for undirected graph
                    edges.append((child_idx, parent_idx))
                    edge_types.append(EdgeType.AST)
        
        # 2. Control flow edges (sequential execution)
        # Simplified: connect nodes in line number order
        sorted_nodes = sorted(ast_nodes, key=lambda n: n.start_line)
        for i in range(len(sorted_nodes) - 1):
            curr_idx = node_mapping[sorted_nodes[i].node_id]
            next_idx = node_mapping[sorted_nodes[i + 1].node_id]
            edges.append((curr_idx, next_idx))
            edge_types.append(EdgeType.CONTROL_FLOW)
        
        # 3. Data flow edges would be added here in a full implementation
        # This requires more sophisticated analysis of variable usage
        
        return edges, edge_types
    
    def _empty_graph(self) -> GraphData:
        """Create an empty graph"""
        return GraphData(
            node_features=torch.zeros((1, self.node_feature_dim)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_type=torch.zeros(0, dtype=torch.long),
            num_nodes=1,
            num_edges=0,
            node_mapping={}
        )
    
    def visualize_graph(self, graph_data: GraphData, output_path: Optional[str] = None):
        """
        Visualize the graph using networkx
        
        Args:
            graph_data: Graph data to visualize
            output_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(graph_data.num_nodes):
            G.add_node(i)
        
        # Add edges
        edges = graph_data.edge_index.t().numpy()
        for edge, edge_type in zip(edges, graph_data.edge_type.numpy()):
            G.add_edge(edge[0], edge[1], type=edge_type)
        
        # Draw
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, arrows=True)
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
        
        plt.close()


# Example usage
if __name__ == "__main__":
    from .ast_parser import parse_code_to_ast
    
    # Test code
    test_code = """
def unsafe_function(user_input):
    import os
    os.system(user_input)
    return True
"""
    
    # Parse AST
    ast_nodes = parse_code_to_ast(test_code, "python")
    
    # Build graph
    builder = GraphBuilder(node_feature_dim=64)
    graph = builder.build_graph(ast_nodes)
    
    print(f"Graph built:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Node features shape: {graph.node_features.shape}")
    print(f"  Edge index shape: {graph.edge_index.shape}")
