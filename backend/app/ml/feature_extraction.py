"""
Script 2: Feature Extraction Module
Production-Ready Code for Senior Project: AI-based Vulnerability Scanner

Purpose: Extract structural (GNN) and sequential (LSTM) features from source code
         using tree-sitter for AST parsing and PyTorch Geometric for graph representation.

Key Components:
- AST/CFG Graph Construction for GNN
- Token Sequence Extraction for LSTM
- PyTorch Geometric Data Format

Author: Senior Project - AI-based Vulnerability Scanner  
Date: 2026-01-25
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import logging

# Tree-sitter imports
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter not available. Install with: pip install tree-sitter tree-sitter-python tree-sitter-javascript")

# PyTorch Geometric imports
try:
    from torch_geometric.data import Data
    import torch_geometric
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logging.warning("PyTorch Geometric not available. Install with: pip install torch-geometric")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphFeatures:
    """Container for graph-based features (for GNN)"""
    node_features: torch.Tensor  # [num_nodes, node_feature_dim]
    edge_index: torch.Tensor     # [2, num_edges]
    edge_attr: Optional[torch.Tensor] = None  # [num_edges, edge_feature_dim]
    node_types: Optional[List[str]] = None
    

@dataclass
class SequenceFeatures:
    """Container for sequence-based features (for LSTM)"""
    token_ids: torch.Tensor      # [seq_length]
    token_embeddings: Optional[torch.Tensor] = None  # [seq_length, embed_dim]
    tokens: Optional[List[str]] = None


class FeatureExtractor:
    """
    Advanced Feature Extraction for Hybrid Deep Learning Model.
    
    Implements:
    1. Structural Analysis (GNN): AST/CFG graph construction
    2. Sequential Analysis (LSTM): Token sequence extraction
    """
    
    # Node type vocabulary for embedding
    NODE_TYPES = [
        # Common across languages
        'program', 'function_definition', 'class_definition',
        'if_statement', 'for_statement', 'while_statement',
        'assignment', 'binary_operation', 'call_expression',
        'identifier', 'string', 'number', 'boolean',
        # Python specific
        'import_statement', 'import_from_statement',
        # JavaScript specific
        'variable_declaration', 'arrow_function', 'method_definition',
        # Control flow
        'return_statement', 'break_statement', 'continue_statement',
        # Special
        'unknown'
    ]
    
    def __init__(self, max_seq_length: int = 512, node_feature_dim: int = 64):
        """
        Initialize feature extractor.
        
        Args:
            max_seq_length: Maximum sequence length for LSTM
            node_feature_dim: Dimension of node feature vectors
        """
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter is required. Install: pip install tree-sitter tree-sitter-python tree-sitter-javascript")
        
        self.max_seq_length = max_seq_length
        self.node_feature_dim = node_feature_dim
        
        # Initialize parsers
        try:
            # New tree-sitter API (v0.22+)
            self.python_parser = Parser(Language(tspython.language()))
            self.javascript_parser = Parser(Language(tsjavascript.language()))
        except (AttributeError, TypeError):
            # Fallback for older versions
            self.python_parser = Parser()
            self.python_parser.set_language(Language(tspython.language()))
            
            self.javascript_parser = Parser()
            self.javascript_parser.set_language(Language(tsjavascript.language()))
        
        # Build vocabulary
        self.node_type_to_idx = {nt: idx for idx, nt in enumerate(self.NODE_TYPES)}
        # Reserve 0 for PAD, 1 for UNK
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_size = 2
        self.vocab_frozen = False
        
        logger.info("Feature Extractor initialized successfully")
    
    def code_to_graph(self, source_code: str, language: str = 'python') -> Data:
        """
        Convert source code to PyTorch Geometric Data object (for GNN).
        
        This implements the STRUCTURAL ANALYSIS component of the hybrid model,
        extracting AST-based graph representations with data flow edges.
        
        Args:
            source_code: Source code string
            language: Programming language ('python' or 'javascript')
            
        Returns:
            PyTorch Geometric Data object with:
            - x: Node feature matrix [num_nodes, node_feature_dim]
            - edge_index: Graph connectivity [2, num_edges]
            - edge_attr: Edge features [num_edges, edge_feature_dim]
        """
        # Parse code to AST
        tree = self._parse_code(source_code, language)
        if tree is None:
            return self._empty_graph()
        
        root_node = tree.root_node
        
        # Extract nodes and edges from AST
        nodes, edges, node_features = self._build_ast_graph(root_node, source_code)
        
        # Add data flow edges (simple version)
        df_edges = self._extract_dataflow_edges(nodes, source_code, language)
        edges.extend(df_edges)
        
        # Convert to PyTorch Geometric format
        pyg_data = self._to_pyg_data(nodes, edges, node_features)
        
        return pyg_data
    
    def code_to_sequence(self, source_code: str, language: str = 'python') -> SequenceFeatures:
        """
        Convert source code to token sequence (for LSTM).
        
        This implements the SEQUENTIAL ANALYSIS component of the hybrid model,
        extracting ordered token sequences for temporal pattern learning.
        
        Args:
            source_code: Source code string
            language: Programming language ('python' or 'javascript')
            
        Returns:
            SequenceFeatures with token IDs and optional embeddings
        """
        # Parse code to AST
        tree = self._parse_code(source_code, language)
        if tree is None:
            return SequenceFeatures(token_ids=torch.zeros(1, dtype=torch.long))
        
        # Extract tokens from AST
        tokens = self._extract_tokens(tree.root_node, source_code)
        
        # Tokenize and convert to IDs
        token_ids = self._tokenize(tokens)
        
        # Truncate or pad to max_seq_length
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        else:
            # Pad with zeros
            padding = [0] * (self.max_seq_length - len(token_ids))
            token_ids = token_ids + padding
        
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        return SequenceFeatures(
            token_ids=token_tensor,
            tokens=tokens[:self.max_seq_length]
        )
    
    def _parse_code(self, source_code: str, language: str) -> Optional[any]:
        """Parse source code using tree-sitter"""
        try:
            code_bytes = source_code.encode('utf-8')
            
            if language == 'python':
                tree = self.python_parser.parse(code_bytes)
            elif language in ['javascript', 'typescript']:
                tree = self.javascript_parser.parse(code_bytes)
            else:
                logger.warning(f"Unsupported language: {language}")
                return None
            
            return tree
        except Exception as e:
            logger.error(f"Parsing error: {e}")
            return None
    
    def _build_ast_graph(
        self, 
        root_node, 
        source_code: str
    ) -> Tuple[List[Dict], List[Tuple[int, int, str]], List[np.ndarray]]:
        """
        Build AST graph from tree-sitter parse tree.
        
        Returns:
            nodes: List of node dicts
            edges: List of (src, dst, edge_type) tuples
            node_features: List of node feature vectors
        """
        nodes = []
        edges = []
        node_features = []
        node_id_map = {}  # tree-sitter node -> graph node ID
        
        def traverse(node, parent_id=None):
            # Create node
            current_id = len(nodes)
            node_type = node.type
            
            # Map to known types
            if node_type not in self.node_type_to_idx:
                node_type = 'unknown'
            
            node_dict = {
                'id': current_id,
                'type': node_type,
                'text': source_code[node.start_byte:node.end_byte][:50]  # Limit text length
            }
            nodes.append(node_dict)
            node_id_map[id(node)] = current_id
            
            # Create node feature vector
            feature_vec = self._create_node_features(node_type, node_dict['text'])
            node_features.append(feature_vec)
            
            # Add edge from parent (AST structure)
            if parent_id is not None:
                edges.append((parent_id, current_id, 'ast'))
            
            # Traverse children
            for child in node.children:
                traverse(child, current_id)
        
        traverse(root_node)
        
        return nodes, edges, node_features
    
    def _create_node_features(self, node_type: str, text: str) -> np.ndarray:
        """
        Create feature vector for a node.
        
        Features include:
        - One-hot encoded node type
        - Text-based features (length, has special chars, etc.)
        """
        features = np.zeros(self.node_feature_dim, dtype=np.float32)
        
        # One-hot encode node type (first len(NODE_TYPES) dimensions)
        type_idx = self.node_type_to_idx.get(node_type, self.node_type_to_idx['unknown'])
        if type_idx < self.node_feature_dim:
            features[type_idx] = 1.0
        
        # Additional features (if space available)
        if self.node_feature_dim > len(self.NODE_TYPES):
            offset = len(self.NODE_TYPES)
            # Text length (normalized)
            features[offset] = min(len(text) / 100.0, 1.0)
            # Has special characters
            features[offset + 1] = float(any(c in text for c in ['$', '*', '&', '|']))
            # Is identifier-like
            features[offset + 2] = float(text.isidentifier() if text else 0)
        
        return features
    
    def _extract_dataflow_edges(
        self,
        nodes: List[Dict],
        source_code: str,
        language: str
    ) -> List[Tuple[int, int, str]]:
        """
        Extract simple data flow edges (variable definitions and uses).
        
        This is a simplified version. Full implementation would require
        more sophisticated program analysis.
        """
        dataflow_edges = []
        
        # Find variable definitions and uses
        definitions = {}  # var_name -> node_id
        
        for i, node in enumerate(nodes):
            node_type = node['type']
            text = node['text']
            
            # Track definitions (assignments)
            if 'assignment' in node_type or 'declaration' in node_type:
                # Extract variable name (simplified)
                var_name = text.split('=')[0].strip() if '=' in text else text
                if var_name:
                    definitions[var_name] = i
            
            # Track uses (identifiers)
            elif node_type == 'identifier' and text in definitions:
                # Add data flow edge from definition to use
                dataflow_edges.append((definitions[text], i, 'dataflow'))
        
        return dataflow_edges
    
    def _to_pyg_data(
        self,
        nodes: List[Dict],
        edges: List[Tuple[int, int, str]],
        node_features: List[np.ndarray]
    ) -> Data:
        """Convert to PyTorch Geometric Data object"""
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric required")
        
        # Node features
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Edge index
        if edges:
            edge_list = [(src, dst) for src, dst, _ in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # Edge attributes (encode edge types)
            edge_types = {'ast': 0, 'dataflow': 1, 'control': 2}
            edge_attr_list = [edge_types.get(edge_type, 0) for _, _, edge_type in edges]
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.long).unsqueeze(1)
        else:
            # Empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _empty_graph(self) -> Data:
        """Create empty graph for error cases"""
        x = torch.zeros((1, self.node_feature_dim), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
    
    def _extract_tokens(self, node, source_code: str) -> List[str]:
        """Extract tokens from AST in depth-first order"""
        tokens = []
        
        def traverse(n):
            # Add node type as token
            tokens.append(n.type)
            
            # If leaf node, add text content
            if len(n.children) == 0:
                text = source_code[n.start_byte:n.end_byte]
                if text.strip():
                    tokens.append(text.strip())
            else:
                # Traverse children
                for child in n.children:
                    traverse(child)
        
        traverse(node)
        return tokens
    
    def _tokenize(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        token_ids = []
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                if not self.vocab_frozen:
                    self.vocab[token] = self.vocab_size
                    self.vocab_size += 1
                    token_ids.append(self.vocab[token])
                else:
                    # Map to UNK (1)
                    token_ids.append(1)
        
        return token_ids
    
    def build_vocabulary(self, code_samples: List[Tuple[str, str]]):
        """
        Build vocabulary from multiple code samples.
        
        Args:
            code_samples: List of (code, language) tuples
        """
        logger.info(f"Building vocabulary from {len(code_samples)} samples...")
        
        all_tokens = []
        for code, language in code_samples:
            try:
                # Use raw parsing to get tokens without modifying vocab yet
                # We need to bypass code_to_sequence which calls _tokenize
                tree = self._parse_code(code, language)
                if tree:
                    tokens = self._extract_tokens(tree.root_node, code)
                    all_tokens.extend(tokens)
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                continue
        
        # Build vocabulary from scratch
        unique_tokens = set(all_tokens)
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for idx, token in enumerate(sorted(unique_tokens)):
            self.vocab[token] = idx + 2
            
        self.vocab_size = len(self.vocab)
        
        logger.info(f"Vocabulary built with {self.vocab_size} tokens")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size for LSTM embedding layer"""
        return self.vocab_size


# Example Usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = FeatureExtractor(max_seq_length=256, node_feature_dim=64)
    
    # Example Python code
    python_code = """
def vulnerable_function(user_input):
    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return execute_query(query)
"""
    
    # Example JavaScript code
    js_code = """
function processPayment(amount) {
    // Command injection vulnerability
    const cmd = `process_payment.sh ${amount}`;
    exec(cmd);
}
"""
    
    print("=" * 80)
    print("STRUCTURAL ANALYSIS (GNN Features)")
    print("=" * 80)
    
    # Extract graph features
    graph_data = extractor.code_to_graph(python_code, language='python')
    print(f"Graph nodes: {graph_data.x.shape[0]}")
    print(f"Graph edges: {graph_data.edge_index.shape[1]}")
    print(f"Node feature dim: {graph_data.x.shape[1]}")
    
    print("\n" + "=" * 80)
    print("SEQUENTIAL ANALYSIS (LSTM Features)")
    print("=" * 80)
    
    # Extract sequence features
    seq_features = extractor.code_to_sequence(python_code, language='python')
    print(f"Sequence length: {len(seq_features.token_ids)}")
    print(f"First 10 tokens: {seq_features.tokens[:10] if seq_features.tokens else 'N/A'}")
    print(f"Vocabulary size: {extractor.get_vocab_size()}")
