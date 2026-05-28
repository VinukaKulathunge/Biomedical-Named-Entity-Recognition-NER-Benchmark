import torch
import torch.nn as nn
import numpy as np

class ELMoEmbedding(nn.Module):
    """
    ELMo (Embeddings from Language Models) contextual embeddings.
    To avoid legacy dependency issues with the deprecated 'allennlp' library on modern
    environments, this class implements an elegant 2-layer Contextual Bidirectional LSTM
    Language Representation layer in PyTorch, which mimics ELMo's architectural behavior
    by producing dynamic token vectors based on their left and right context.
    """
    def __init__(self, sentences=None, word_dim: int = 128, lstm_hidden: int = 128):
        super().__init__()
        # Total output dimension = 2 * lstm_hidden (since it is bidirectional) = 256
        self.dim = 2 * lstm_hidden
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        
        # Build vocabulary from sentences
        if sentences is not None:
            for sent in sentences:
                for token in sent:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        
        # 1. Base Token Representation Layer
        self.token_embedding = nn.Embedding(
            len(self.vocab), 
            word_dim, 
            padding_idx=0
        )
        
        # 2. 2-layer Bidirectional LSTM Contextualizer (ELMo Core)
        self.contextualizer = nn.LSTM(
            input_size=word_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        print(f"  [ELMo] Contextual BiLSTM initialized. Vocab size: {len(self.vocab)}. Dim: {self.dim}")

    def get_dim(self) -> int:
        return self.dim
        
    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Extracts contextualized representations.
        
        Args:
            token_indices (torch.Tensor): Integer indices of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Contextual vectors of shape (batch_size, seq_len, 2 * lstm_hidden)
        """
        # Step 1: Base static embeddings -> (batch_size, seq_len, word_dim)
        embedded = self.token_embedding(token_indices)
        
        # Step 2: Pass through 2-layer BiLSTM to gather contextual states
        # outputs shape: (batch_size, seq_len, 2 * lstm_hidden)
        outputs, _ = self.contextualizer(embedded)
        
        return outputs
        
    def text_to_indices(self, tokens_list: list) -> list:
        return [self.vocab.get(token, 1) for token in tokens_list]
