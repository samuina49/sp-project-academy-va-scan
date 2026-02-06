"""
LSTM Model for Sequential Code Analysis

This module implements a bidirectional LSTM network for processing
tokenized code sequences.

Architecture:
- Token embedding layer
- Bidirectional LSTM layers
- Optionally with attention mechanism
- Outputs fixed-size sequence representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LSTMModel(nn.Module):
    """
    LSTM network for code vulnerability detection
    
    Takes tokenized code sequences and produces a fixed-size
    embedding that captures sequential patterns.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = False
    ):
        """
        Initialize LSTM model
        
        Args:
            vocab_size: Size of token vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output embedding dimension
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
        """
        super(LSTMModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # Assuming 0 is PAD token
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Output projection
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through LSTM
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Mask for padded tokens [batch_size, seq_length]
        
        Returns:
            Tuple of (sequence_embedding [batch_size, output_dim], 
                     attention_weights [batch_size, seq_length] or None)
        """
        batch_size, seq_length = input_ids.shape
        
        # Embed tokens
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        embedded = self.dropout_layer(embedded)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: [batch, seq_len, hidden_dim * 2] if bidirectional
        
        if self.use_attention:
            # Attention mechanism
            attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attention_weights = attention_weights.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, 
                    float('-inf')
                )
            
            # Softmax to get attention scores
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Weighted sum of LSTM outputs
            context = torch.sum(attention_weights * lstm_out, dim=1)
            # context: [batch, hidden_dim * 2]
            
            attention_weights = attention_weights.squeeze(-1)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward last hidden states
                hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
                context = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=1)
            else:
                context = hidden[-1]
            # context: [batch, hidden_dim * 2] or [batch, hidden_dim]
            
            attention_weights = None
        
        # Project to output dimension
        output = self.fc(context)  # [batch, output_dim]
        output = self.dropout_layer(output)
        
        return output, attention_weights
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.output_dim


# Example usage
if __name__ == "__main__":
    # Sample parameters
    vocab_size = 10000
    batch_size = 8
    seq_length = 512
    
    # Random token IDs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones(batch_size, seq_length)
    attention_mask[:, 400:] = 0  # Last 112 tokens are padding
    
    # Create model
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        output_dim=128,
        dropout=0.3,
        bidirectional=True,
        use_attention=False
    )
    
    # Forward pass
    output, attn_weights = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dimension: {model.get_output_dim()}")
    
    # Test with attention
    model_attn = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        output_dim=128,
        dropout=0.3,
        bidirectional=True,
        use_attention=True
    )
    
    output_attn, attn_weights = model_attn(input_ids, attention_mask)
    print(f"\nWith attention:")
    print(f"Output shape: {output_attn.shape}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
