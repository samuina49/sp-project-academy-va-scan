"""
AST Parser for Multi-Language Support

This module parses source code into Abstract Syntax Trees (AST)
using tree-sitter for Python, JavaScript, and TypeScript.

The AST is used as input to the Graph Neural Network.
"""

import ast
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


@dataclass
class ASTNode:
    """Represents a node in the Abstract Syntax Tree"""
    node_id: int
    node_type: str
    value: Optional[str]
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    parent_id: Optional[int]
    children: List[int]
    attributes: Dict[str, Any]


class ASTParser:
    """
    Parse source code into Abstract Syntax Trees
    
    This uses Python's built-in ast module for Python code.
    For production, you would use tree-sitter for all languages.
    """
    
    def __init__(self, language: Language = Language.PYTHON):
        """
        Initialize AST parser
        
        Args:
            language: Programming language to parse
        """
        self.language = language
        self.node_counter = 0
        self.nodes: Dict[int, ASTNode] = {}
        
    def parse(self, code: str) -> List[ASTNode]:
        """
        Parse source code into AST nodes
        
        Args:
            code: Source code string
            
        Returns:
            List of ASTNode objects representing the tree
        """
        self.reset()
        
        if self.language == Language.PYTHON:
            return self._parse_python(code)
        elif self.language == Language.JAVASCRIPT:
            return self._parse_javascript(code)
        elif self.language == Language.TYPESCRIPT:
            return self._parse_typescript(code)
        else:
            raise ValueError(f"Unsupported language: {self.language}")
    
    def reset(self):
        """Reset parser state"""
        self.node_counter = 0
        self.nodes = {}
    
    def _parse_python(self, code: str) -> List[ASTNode]:
        """
        Parse Python code using ast module
        
        Args:
            code: Python source code
            
        Returns:
            List of AST nodes
        """
        try:
            tree = ast.parse(code)
            self._visit_python_node(tree, parent_id=None)
            return list(self.nodes.values())
        except SyntaxError as e:
            # Return empty list for malformed code
            print(f"Syntax error in code: {e}")
            return []
    
    def _visit_python_node(self, node: ast.AST, parent_id: Optional[int] = None) -> int:
        """
        Recursively visit Python AST nodes
        
        Args:
            node: Python AST node
            parent_id: Parent node ID
            
        Returns:
            Current node ID
        """
        node_id = self.node_counter
        self.node_counter += 1
        
        # Extract node information
        node_type = node.__class__.__name__
        value = None
        attributes = {}
        
        # Extract specific attributes based on node type
        if isinstance(node, ast.Name):
            value = node.id
        elif isinstance(node, ast.Str):
            value = node.s[:50]  # Truncate long strings
        elif isinstance(node, ast.Num):
            value = str(node.n)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            value = node.name
            attributes['name'] = node.name
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                value = node.func.id
                attributes['function'] = node.func.id
        
        # Get position information
        lineno = getattr(node, 'lineno', 0)
        col_offset = getattr(node, 'col_offset', 0)
        end_lineno = getattr(node, 'end_lineno', lineno)
        end_col_offset = getattr(node, 'end_col_offset', col_offset)
        
        # Create AST node
        ast_node = ASTNode(
            node_id=node_id,
            node_type=node_type,
            value=value,
            start_line=lineno,
            end_line=end_lineno,
            start_col=col_offset,
            end_col=end_col_offset,
            parent_id=parent_id,
            children=[],
            attributes=attributes
        )
        
        self.nodes[node_id] = ast_node
        
        # Update parent's children list
        if parent_id is not None:
            self.nodes[parent_id].children.append(node_id)
        
        # Visit children
        for child in ast.iter_child_nodes(node):
            self._visit_python_node(child, parent_id=node_id)
        
        return node_id
    
    def _parse_javascript(self, code: str) -> List[ASTNode]:
        """
        Parse JavaScript code
        
        Note: This is a placeholder. For production, use tree-sitter.
        
        Args:
            code: JavaScript source code
            
        Returns:
            List of AST nodes
        """
        # Placeholder - would use tree-sitter in production
        print("JavaScript parsing not yet implemented. Use tree-sitter.")
        return []
    
    def _parse_typescript(self, code: str) -> List[ASTNode]:
        """
        Parse TypeScript code
        
        Note: This is a placeholder. For production, use tree-sitter.
        
        Args:
            code: TypeScript source code
            
        Returns:
            List of AST nodes
        """
        # Placeholder - would use tree-sitter in production
        print("TypeScript parsing not yet implemented. Use tree-sitter.")
        return []
    
    def get_node_features(self, node: ASTNode) -> Dict[str, Any]:
        """
        Extract features from an AST node for ML model
        
        Args:
            node: AST node
            
        Returns:
            Dictionary of features
        """
        return {
            'node_type': node.node_type,
            'has_value': node.value is not None,
            'depth': self._calculate_depth(node.node_id),
            'num_children': len(node.children),
            'line_span': node.end_line - node.start_line + 1,
        }
    
    def _calculate_depth(self, node_id: int) -> int:
        """Calculate depth of node in tree"""
        depth = 0
        current_id = node_id
        
        while current_id is not None:
            node = self.nodes.get(current_id)
            if node is None or node.parent_id is None:
                break
            current_id = node.parent_id
            depth += 1
        
        return depth


def parse_code_to_ast(code: str, language: str = "python") -> List[ASTNode]:
    """
    Convenience function to parse code to AST
    
    Args:
        code: Source code string
        language: Programming language (python, javascript, typescript)
        
    Returns:
        List of AST nodes
    """
    lang_enum = Language(language.lower())
    parser = ASTParser(language=lang_enum)
    return parser.parse(code)


# Example usage
if __name__ == "__main__":
    # Test with vulnerable Python code
    test_code = """
import os

def unsafe_command(user_input):
    # Vulnerable: command injection
    os.system("echo " + user_input)

def unsafe_sql(username):
    # Vulnerable: SQL injection
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return query
"""
    
    parser = ASTParser(Language.PYTHON)
    nodes = parser.parse(test_code)
    
    print(f"Parsed {len(nodes)} AST nodes")
    for node in nodes[:10]:  # Print first 10
        print(f"  {node.node_type}: {node.value} (line {node.start_line})")
