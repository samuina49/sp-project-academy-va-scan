"""
Code Tokenizer for LSTM Input

This module tokenizes source code into sequences of tokens
for processing by the LSTM network.

Supports:
- Word-level tokenization
- Byte-Pair Encoding (BPE)
- Special tokens for code constructs
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from collections import Counter
import torch


class CodeTokenizer:
    """
    Tokenize source code for LSTM processing
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"  # Beginning of sequence
    EOS_TOKEN = "<EOS>"  #End of sequence
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 512):
        """
        Initialize tokenizer
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_base_vocab()
        
    def _build_base_vocab(self):
        """Build base vocabulary with special tokens"""
        special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN
        ]
        
        for idx, token in enumerate(special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def train(self, code_samples: List[str]):
        """
        Build vocabulary from code samples
        
        Args:
            code_samples: List of code strings
        """
        # Tokenize all samples
        all_tokens = []
        for code in code_samples:
            tokens = self._tokenize_code(code)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Get most common tokens
        most_common = token_counts.most_common(self.vocab_size - len(self.token_to_id))
        
        # Add to vocabulary
        current_id = len(self.token_to_id)
        for token, _ in most_common:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
    
    def _tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize code into list of tokens
        
        Args:
            code: Source code string
            
        Returns:
            List of tokens
        """
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # JS comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Multi-line comments
        
        # Tokenize by splitting on whitespace and operators
        # Keep operators as separate tokens
        pattern = r'(\b\w+\b|[^\w\s])'
        tokens = re.findall(pattern, code)
        
        # Filter empty tokens
        tokens = [t.strip() for t in tokens if t.strip()]
        
        return tokens
    
    def encode(self, code: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode code to list of token IDs
        
        Args:
            code: Source code string
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self._tokenize_code(code)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.BOS_TOKEN] + tokens + [self.EOS_TOKEN]
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.EOS_TOKEN]
        
        # Convert to IDs
        token_ids = [
            self.token_to_id.get(token, self.token_to_id[self.UNK_TOKEN])
            for token in tokens
        ]
        
        return token_ids
    
    def encode_batch(
        self, 
        code_samples: List[str], 
        padding: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode batch of code samples
        
        Args:
            code_samples: List of code strings
            padding: Whether to pad sequences
            
        Returns:
            Tuple of (token_ids, attention_mask)
        """
        # Encode all samples
        encoded = [self.encode(code) for code in code_samples]
        
        if padding:
            # Pad sequences
            max_len = max(len(seq) for seq in encoded)
            max_len = min(max_len, self.max_length)
            
            padded = []
            masks = []
            
            for seq in encoded:
                # Pad or truncate
                if len(seq) < max_len:
                    mask = [1] * len(seq) + [0] * (max_len - len(seq))
                    seq = seq + [self.token_to_id[self.PAD_TOKEN]] * (max_len - len(seq))
                else:
                    mask = [1] * max_len
                    seq = seq[:max_len]
                
                padded.append(seq)
                masks.append(mask)
            
            return (
                torch.tensor(padded, dtype=torch.long),
                torch.tensor(masks, dtype=torch.long)
            )
        else:
            return torch.tensor(encoded, dtype=torch.long), None
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to code
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded code string
        """
        tokens = [
            self.id_to_token.get(tid, self.UNK_TOKEN)
            for tid in token_ids
        ]
        
        # Remove special tokens
        tokens = [
            t for t in tokens 
            if t not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        ]
        
        # Join tokens
        code = ' '.join(tokens)
        
        return code
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.max_length = vocab_data['max_length']
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size"""
        return len(self.token_to_id)


# Example usage
if __name__ == "__main__":
    # Sample vulnerable code
    code_samples = [
        """
import os
def unsafe_cmd(user_input):
    os.system("echo " + user_input)
        """,
        """
def sql_injection(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return query
        """,
        """
def eval_vuln(user_code):
    result = eval(user_code)
    return result
        """
    ]
    
    # Create and train tokenizer
    tokenizer = CodeTokenizer(vocab_size=1000, max_length=128)
    tokenizer.train(code_samples)
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Encode sample
    encoded = tokenizer.encode(code_samples[0])
    print(f"\nEncoded length: {len(encoded)}")
    print(f"First 10 tokens: {encoded[:10]}")
    
    # Decode back
    decoded = tokenizer.decode(encoded)
    print(f"\nDecoded (first 100 chars): {decoded[:100]}")
    
    # Batch encoding
    token_ids, masks = tokenizer.encode_batch(code_samples)
    print(f"\nBatch shape: {token_ids.shape}")
    print(f"Mask shape: {masks.shape}")
