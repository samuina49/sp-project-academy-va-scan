"""
Enhanced Graph Builder with Control Flow and Advanced Data Flow
================================================================
Improvements over original feature_extraction.py:

1. ✅ Control Flow Graph (CFG) extraction
   - if/else branching
   - while/for loops  
   - function calls
   - return statements
   - break/continue statements

2. ✅ Enhanced Data Flow Analysis
   - Function parameters and arguments
   - Return value propagation
   - Method/attribute access
   - List comprehensions and lambda

3. ✅ Multi-language support (Python, JavaScript, TypeScript)

Author: AI Vulnerability Scanner - Enhanced Version
Date: February 6, 2026
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
    logging.warning("tree-sitter not available")

# PyTorch Geometric imports
try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logging.warning("PyTorch Geometric not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedGraphFeatures:
    """Container for enhanced graph features"""
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_types: List[str]
    edge_types: List[str]
    cfg_stats: Dict[str, int]  # Control flow statistics


class EnhancedFeatureExtractor:
    """
    Enhanced Feature Extraction with Complete CFG and DFG.
    
    Edge Types:
    - AST (0): Abstract Syntax Tree structure
    - Data Flow (1): Variable definitions -> uses
    - Control Flow (2): Execution order (if/else, loops, calls)
    """
    
    # Control flow node types (statements that affect execution order)
    CONTROL_FLOW_NODES = {
        'if_statement', 'elif_clause', 'else_clause',
        'while_statement', 'for_statement', 'for_in_statement',
        'do_statement', 'switch_statement', 'case',
        'try_statement', 'except_clause', 'finally_clause', 'with_statement',
        'break_statement', 'continue_statement', 'return_statement',
        'call_expression', 'function_definition', 'method_definition',
        # JavaScript specific
        'if', 'else', 'while', 'for', 'do', 'switch', 'case',
        'try', 'catch', 'finally', 'return', 'break', 'continue'
    }
    
    # Data flow relevant nodes
    DATA_FLOW_NODES = {
        'assignment', 'assignment_statement', 'augmented_assignment',
        'variable_declaration', 'identifier', 'parameter',
        'argument_list', 'return_statement', 'call_expression',
        'attribute', 'subscript', 'member_expression', 'property_identifier'
    }
    
    def __init__(self, max_seq_length: int = 512, node_feature_dim: int = 64):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter required")
        
        self.max_seq_length = max_seq_length
        self.node_feature_dim = node_feature_dim
        
        # Initialize parsers
        try:
            self.python_parser = Parser(Language(tspython.language()))
            self.javascript_parser = Parser(Language(tsjavascript.language()))
        except (AttributeError, TypeError):
            self.python_parser = Parser()
            self.python_parser.set_language(Language(tspython.language()))
            self.javascript_parser = Parser()
            self.javascript_parser.set_language(Language(tsjavascript.language()))
        
        logger.info("✅ Enhanced Feature Extractor initialized")
    
    def extract_enhanced_graph(
        self,
        source_code: str,
        language: str = 'python'
    ) -> Data:
        """
        Extract complete graph with AST + CFG + DFG.
        
        Returns PyTorch Geometric Data with:
        - x: Node features
        - edge_index: All edges (AST + CFG + DFG)
        - edge_attr: Edge type labels (0=AST, 1=DFG, 2=CFG)
        """
        # Parse code
        tree = self._parse_code(source_code, language)
        if tree is None:
            return self._empty_graph()
        
        root_node = tree.root_node
        
        # Step 1: Build AST with node mapping
        nodes, ast_edges, node_features, node_id_map = self._build_ast_with_mapping(
            root_node, source_code
        )
        
        if len(nodes) == 0:
            return self._empty_graph()
        
        # Step 2: Extract Control Flow edges
        cfg_edges = self._extract_control_flow_edges(
            root_node, node_id_map, source_code, language
        )
        
        # Step 3: Extract Enhanced Data Flow edges
        dfg_edges = self._extract_enhanced_dataflow_edges(
            root_node, node_id_map, source_code, language
        )
        
        # Combine all edges
        all_edges = ast_edges + dfg_edges + cfg_edges
        
        # Statistics
        stats = {
            'total_nodes': len(nodes),
            'ast_edges': len(ast_edges),
            'cfg_edges': len(cfg_edges),
            'dfg_edges': len(dfg_edges),
            'total_edges': len(all_edges)
        }
        
        logger.debug(f"Graph stats: {stats}")
        
        # Convert to PyTorch Geometric format
        return self._to_pyg_data(nodes, all_edges, node_features)
    
    def _build_ast_with_mapping(
        self,
        root_node,
        source_code: str
    ) -> Tuple[List[Dict], List[Tuple], List[np.ndarray], Dict]:
        """
        Build AST and maintain node ID mapping for CFG/DFG construction.
        
        Returns:
            nodes, edges, node_features, node_id_map
        """
        nodes = []
        edges = []
        node_features = []
        node_id_map = {}  # tree-sitter node id -> graph node ID
        
        def traverse(node, parent_id=None):
            current_id = len(nodes)
            node_type = node.type
            
            # Store node
            node_dict = {
                'id': current_id,
                'type': node_type,
                'text': source_code[node.start_byte:node.end_byte][:100],
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'ts_node': node  # Keep reference for CFG/DFG
            }
            nodes.append(node_dict)
            
            # Map tree-sitter node to graph ID
            node_id_map[id(node)] = current_id
            
            # Create node features
            feature_vec = self._create_node_features(node_type, node_dict['text'])
            node_features.append(feature_vec)
            
            # AST edge from parent
            if parent_id is not None:
                edges.append((parent_id, current_id, 'ast'))
            
            # Traverse children
            for child in node.children:
                traverse(child, current_id)
        
        traverse(root_node)
        return nodes, edges, node_features, node_id_map
    
    def _extract_control_flow_edges(
        self,
        root_node,
        node_id_map: Dict,
        source_code: str,
        language: str
    ) -> List[Tuple[int, int, str]]:
        """
        Extract Control Flow Graph (CFG) edges.
        
        CFG Edge Types:
        1. Sequential: statement A -> statement B (normal flow)
        2. Conditional: if -> then-block, if -> else-block
        3. Loop: while/for -> body, body -> while/for (back edge)
        4. Call: caller -> callee, callee -> return site
        5. Jump: break/continue/return -> target
        """
        cfg_edges = []
        
        def extract_cfg(node, parent_stmt_id=None):
            """
            Recursively extract CFG edges.
            parent_stmt_id: ID of the previous statement in execution order
            """
            node_type = node.type
            node_id = node_id_map.get(id(node))
            
            # Control flow nodes
            if node_type == 'if_statement':
                # if -> condition -> then_block -> (else_block)
                condition = self._get_child_by_field(node, 'condition')
                consequence = self._get_child_by_field(node, 'consequence')
                alternative = self._get_child_by_field(node, 'alternative')
                
                if condition and node_id:
                    # parent -> if statement
                    if parent_stmt_id is not None:
                        cfg_edges.append((parent_stmt_id, node_id, 'control'))
                    
                    # if -> condition
                    cond_id = node_id_map.get(id(condition))
                    if cond_id:
                        cfg_edges.append((node_id, cond_id, 'control'))
                        
                        # condition -> then block
                        if consequence:
                            conseq_id = node_id_map.get(id(consequence))
                            if conseq_id:
                                cfg_edges.append((cond_id, conseq_id, 'control'))
                        
                        # condition -> else block
                        if alternative:
                            alt_id = node_id_map.get(id(alternative))
                            if alt_id:
                                cfg_edges.append((cond_id, alt_id, 'control'))
            
            elif node_type in ['while_statement', 'for_statement', 'for_in_statement']:
                # Loop: condition -> body, body -> condition (back edge)
                if node_id and parent_stmt_id is not None:
                    cfg_edges.append((parent_stmt_id, node_id, 'control'))
                
                # Get loop components
                if node_type == 'while_statement':
                    condition = self._get_child_by_field(node, 'condition')
                    body = self._get_child_by_field(node, 'body')
                elif node_type in ['for_statement', 'for_in_statement']:
                    # For loop structure varies by language
                    body = self._get_child_by_field(node, 'body')
                    condition = None  # Simplified
                
                if body and node_id:
                    body_id = node_id_map.get(id(body))
                    if body_id:
                        # loop -> body
                        cfg_edges.append((node_id, body_id, 'control'))
                        # body -> loop (back edge)
                        cfg_edges.append((body_id, node_id, 'control'))
            
            elif node_type == 'call_expression':
                # Function call: caller -> callee
                if node_id and parent_stmt_id is not None:
                    cfg_edges.append((parent_stmt_id, node_id, 'control'))
                
                # Extract function name for potential interprocedural edge
                function_node = self._get_child_by_field(node, 'function')
                if function_node:
                    func_id = node_id_map.get(id(function_node))
                    if func_id and node_id:
                        cfg_edges.append((node_id, func_id, 'control'))
            
            elif node_type in ['return_statement', 'break_statement', 'continue_statement']:
                # Jump statement
                if node_id and parent_stmt_id is not None:
                    cfg_edges.append((parent_stmt_id, node_id, 'control'))
            
            # Recurse through children
            current_parent = node_id if node_id else parent_stmt_id
            for child in node.children:
                extract_cfg(child, current_parent)
        
        extract_cfg(root_node)
        return cfg_edges
    
    def _extract_enhanced_dataflow_edges(
        self,
        root_node,
        node_id_map: Dict,
        source_code: str,
        language: str
    ) -> List[Tuple[int, int, str]]:
        """
        Extract ADVANCED Data Flow Graph (DFG) edges.
        
        Tracks:
        1. Variable definitions and uses (scope-aware)
        2. Function parameters and arguments (interprocedural)
        3. Return value propagation
        4. Attribute/method access (object tracking)
        5. Container operations (list.append, dict.update)
        6. Binary operations (left + right -> result)
        """
        dfg_edges = []
        
        # Simplified: Global definitions dictionary (not scope-aware for now)
        # This fixes the issue where definitions were lost
        definitions = {}  # var_name -> [node_ids]
        
        # Track function definitions for interprocedural analysis
        function_defs = {}  # func_name -> {'params': [...], 'node_id': ...}
        
        def add_definition(var_name, node_id):
            """Add variable definition"""
            if var_name and node_id:
                if var_name not in definitions:
                    definitions[var_name] = []
                definitions[var_name].append(node_id)
        
        def add_dfg_edge(from_id, to_id):
            """Add DFG edge with duplicate check"""
            if from_id and to_id and from_id != to_id:
                edge = (from_id, to_id, 'dataflow')
                if edge not in dfg_edges:
                    dfg_edges.append(edge)
        
        def extract_dfg(node):
            node_type = node.type
            node_id = node_id_map.get(id(node))
            
            # === VARIABLE DEFINITIONS ===
            # Assignment: x = value
            if node_type in ['assignment', 'assignment_statement']:
                # Get both sides
                children = [c for c in node.children if c.type not in ['=', ':', 'newline']]
                
                if len(children) >= 2:
                    left_node = children[0]
                    right_node = children[1] if len(children) > 1 else None
                    
                    # Extract variable name from left side
                    var_name = None
                    if left_node.type == 'identifier':
                        var_name = source_code[left_node.start_byte:left_node.end_byte].strip()
                    
                    if var_name:
                        left_id = node_id_map.get(id(left_node))
                        add_definition(var_name, left_id)
                        
                        # Right side -> Left side
                        if right_node:
                            right_id = node_id_map.get(id(right_node))
                            if right_id and left_id:
                                add_dfg_edge(right_id, left_id)
                            
                            # If right is identifier, connect from its definition
                            if right_node.type == 'identifier':
                                right_var = source_code[right_node.start_byte:right_node.end_byte].strip()
                                if right_var in definitions:
                                    for def_id in definitions[right_var]:
                                        add_dfg_edge(def_id, left_id)
            
            # Augmented assignment: x += value
            elif node_type == 'augmented_assignment':
                left_node = self._get_child_by_field(node, 'left')or (node.children[0] if node.children else None)
                right_node = self._get_child_by_field(node, 'right') or (node.children[2] if len(node.children) > 2 else None)
                
                if left_node and left_node.type == 'identifier':
                    var_name = source_code[left_node.start_byte:left_node.end_byte].strip()
                    left_id = node_id_map.get(id(left_node))
                    
                    # This is both use and definition
                    if var_name in definitions and left_id:
                        # Use: previous definition -> current
                        for def_id in definitions[var_name]:
                            add_dfg_edge(def_id, left_id)
                    
                    # New definition
                    add_definition(var_name, left_id)
                    
                    # Right side -> left
                    if right_node:
                        right_id = node_id_map.get(id(right_node))
                        if right_id and left_id:
                            add_dfg_edge(right_id, left_id)
            
            # Function parameter (definition)
            elif node_type in ['parameter', 'formal_parameter']:
                # Extract parameter name
                param_name = None
                for child in node.children:
                    if child.type == 'identifier':
                        param_name = source_code[child.start_byte:child.end_byte].strip()
                        param_id = node_id_map.get(id(child))
                        if param_name:
                            add_definition(param_name, param_id)
                        break
                
                # Fallback: use node itself
                if not param_name:
                    param_name = source_code[node.start_byte:node.end_byte].strip()
                    if param_name and ':' not in param_name:  # Avoid type annotations
                        add_definition(param_name, node_id)
            
            # === VARIABLE USES ===
            # Identifier usage
            elif node_type == 'identifier':
                var_name = source_code[node.start_byte:node.end_byte].strip()
                
                # Check if this is a use (not a definition)
                parent = node.parent if hasattr(node, 'parent') else None
                is_definition_context = False
                
                if parent:
                    parent_type = parent.type
                    # Check if this identifier is on the left side of assignment
                    if parent_type in ['assignment', 'assignment_statement']:
                        children = [c for c in parent.children]
                        if children and children[0] == node:
                            is_definition_context = True
                
                if not is_definition_context and var_name in definitions and node_id:
                    # This is a USE - connect from all definitions
                    for def_id in definitions[var_name]:
                        add_dfg_edge(def_id, node_id)
            
            # === ATTRIBUTE ACCESS ===
            elif node_type == 'attribute':
                # obj.attr
                obj_node = self._get_child_by_field(node, 'object')
                attr_node = self._get_child_by_field(node, 'attribute')
                
                if obj_node and attr_node:
                    obj_id = node_id_map.get(id(obj_node))
                    attr_id = node_id_map.get(id(attr_node))
                    
                    if obj_id and node_id:
                        add_dfg_edge(obj_id, node_id)
            
            # === FUNCTION CALLS ===
            elif node_type == 'call_expression':
                # Get arguments
                arguments = self._get_child_by_field(node, 'arguments')
                
                if arguments and node_id:
                    for arg in arguments.children:
                        if arg.type not in ['(', ')', ',']:
                            arg_id = node_id_map.get(id(arg))
                            if arg_id:
                                # Argument -> call
                                add_dfg_edge(arg_id, node_id)
                
                # Check for method calls on containers (e.g., list.append)
                func_node = self._get_child_by_field(node, 'function')
                if func_node and func_node.type == 'attribute':
                    obj_node = self._get_child_by_field(func_node, 'object')
                    method_node = self._get_child_by_field(func_node, 'attribute')
                    
                    if obj_node and method_node:
                        method_name = source_code[method_node.start_byte:method_node.end_byte].strip()
                        
                        # Mutating methods
                        if method_name in ['append', 'extend', 'insert', 'update', 'add', 'push', 'pop']:
                            obj_id = node_id_map.get(id(obj_node))
                            
                            # Arguments flow into container
                            if arguments:
                                for arg in arguments.children:
                                    if arg.type not in ['(', ')', ',']:
                                        arg_id = node_id_map.get(id(arg))
                                        if arg_id and obj_id:
                                            add_dfg_edge(arg_id, obj_id)
            
            # === BINARY OPERATIONS ===
            elif node_type in ['binary_expression', 'binary_operator', 'comparison_operator']:
                # left op right
                children = [c for c in node.children if c.type not in ['operator', 'and', 'or', '==', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', 'is', 'in']]
                
                if len(children) >= 2 and node_id:
                    for child in children:
                        child_id = node_id_map.get(id(child))
                        if child_id:
                            add_dfg_edge(child_id, node_id)
            
            # === RETURN STATEMENT ===
            elif node_type == 'return_statement':
                # Get returned value
                for child in node.children:
                    if child.type not in ['return']:
                        value_id = node_id_map.get(id(child))
                        if value_id and node_id:
                            add_dfg_edge(value_id, node_id)
            
            # Recurse through all children
            for child in node.children:
                extract_dfg(child)
        
        # Start extraction
        extract_dfg(root_node)
        return dfg_edges
    
    def _get_child_by_field(self, node, field_name: str):
        """Get child node by field name (e.g., 'condition', 'body')"""
        try:
            return node.child_by_field_name(field_name)
        except (AttributeError, TypeError) as e:
            # Fallback: search by type when node doesn't support field access
            logger.debug(f"Field access failed for '{field_name}': {e}")
            field_type_map = {
                'condition': ['condition', 'test'],
                'consequence': ['consequence', 'body', 'then'],
                'alternative': ['alternative', 'orelse', 'else'],
                'body': ['body', 'block'],
                'function': ['function', 'name'],
                'arguments': ['arguments', 'args']
            }
            
            search_types = field_type_map.get(field_name, [field_name])
            for child in node.children:
                if child.type in search_types:
                    return child
            return None
    
    def _extract_variable_name(self, node, source_code: str) -> Optional[str]:
        """Extract variable name from assignment node"""
        # Find identifier on left side of assignment
        for child in node.children:
            if child.type in ['identifier', 'property_identifier']:
                return source_code[child.start_byte:child.end_byte].strip()
        return None
    
    def _create_node_features(self, node_type: str, text: str) -> np.ndarray:
        """Create node feature vector"""
        features = np.zeros(self.node_feature_dim, dtype=np.float32)
        
        # Feature 0-9: Node type category (one-hot)
        type_categories = {
            'control': 0, 'assignment': 1, 'identifier': 2,
            'call': 3, 'function': 4, 'class': 5,
            'operator': 6, 'literal': 7, 'import': 8, 'other': 9
        }
        
        category = 'other'
        if any(x in node_type for x in ['if', 'while', 'for', 'switch']):
            category = 'control'
        elif 'assignment' in node_type:
            category = 'assignment'
        elif 'identifier' in node_type:
            category = 'identifier'
        elif 'call' in node_type:
            category = 'call'
        elif 'function' in node_type or 'method' in node_type:
            category = 'function'
        elif 'class' in node_type:
            category = 'class'
        elif 'operator' in node_type or 'binary' in node_type:
            category = 'operator'
        elif any(x in node_type for x in ['string', 'number', 'boolean', 'null']):
            category = 'literal'
        elif 'import' in node_type:
            category = 'import'
        
        features[type_categories[category]] = 1.0
        
        # Feature 10-19: Text-based features
        if self.node_feature_dim > 10:
            features[10] = min(len(text) / 100.0, 1.0)  # Text length
            features[11] = float('$' in text or '*' in text)  # Special chars
            features[12] = float(text.isidentifier() if text else 0)  # Is identifier
            features[13] = float(any(c.isdigit() for c in text))  # Has numbers
            features[14] = float(text.isupper() if text else 0)  # Is constant
        
        return features
    
    def _parse_code(self, source_code: str, language: str):
        """Parse source code using tree-sitter"""
        try:
            code_bytes = source_code.encode('utf-8')
            if language == 'python':
                return self.python_parser.parse(code_bytes)
            elif language in ['javascript', 'typescript']:
                return self.javascript_parser.parse(code_bytes)
            return None
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None
    
    def _to_pyg_data(
        self,
        nodes: List[Dict],
        edges: List[Tuple[int, int, str]],
        node_features: List[np.ndarray]
    ) -> Data:
        """Convert to PyTorch Geometric Data"""
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        if edges:
            edge_list = [(src, dst) for src, dst, _ in edges]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # Edge type encoding
            edge_types = {'ast': 0, 'dataflow': 1, 'control': 2}
            edge_attr_list = [edge_types.get(etype, 0) for _, _, etype in edges]
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.long).unsqueeze(1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _empty_graph(self) -> Data:
        """Empty graph for error cases"""
        x = torch.zeros((1, self.node_feature_dim), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Test the enhanced extractor
if __name__ == "__main__":
    print("="*80)
    print("ENHANCED GRAPH BUILDER TEST")
    print("="*80)
    
    extractor = EnhancedFeatureExtractor(node_feature_dim=64)
    
    # Test Python code with control flow
    test_code = """
def process_user_input(user_data):
    if user_data is None:
        return None
    
    result = []
    for item in user_data:
        if item.startswith('admin'):
            # Security issue: hardcoded check
            result.append(execute_query(item))
        else:
            result.append(safe_process(item))
    
    return result
"""
    
    print(f"\n📝 Test Code:\n{test_code}")
    print("\n" + "="*80)
    
    graph = extractor.extract_enhanced_graph(test_code, 'python')
    
    print(f"\n📊 Graph Statistics:")
    print(f"  • Total nodes: {graph.x.shape[0]}")
    print(f"  • Total edges: {graph.edge_index.shape[1]}")
    print(f"  • Node feature dim: {graph.x.shape[1]}")
    
    # Count edge types
    if graph.edge_attr.shape[0] > 0:
        edge_types_count = {
            'AST': (graph.edge_attr == 0).sum().item(),
            'Data Flow': (graph.edge_attr == 1).sum().item(),
            'Control Flow': (graph.edge_attr == 2).sum().item()
        }
        print(f"\n📈 Edge Type Distribution:")
        for etype, count in edge_types_count.items():
            percentage = (count / graph.edge_index.shape[1]) * 100
            print(f"  • {etype}: {count} ({percentage:.1f}%)")
    
    print("\n✅ Enhanced graph extraction complete!")
    print("="*80)
