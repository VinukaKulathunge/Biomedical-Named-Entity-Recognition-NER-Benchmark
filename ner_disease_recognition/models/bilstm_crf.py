import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF sequence labeling model.
    Encodes dense word embeddings using a bidirectional LSTM, maps the output
    to tag space, and feeds these emission scores to a Conditional Random Field (CRF)
    layer to decode the globally optimal sequence of labels.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, num_tags: int, dropout_rate: float = 0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        
        # Bidirectional LSTM sequence encoder
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense projection layer mapping BiLSTM outputs to tag emissions
        # Hidden dimension is multiplied by 2 because the LSTM is bidirectional
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)
        
        # CRF layer for global transition scoring and Viterbi decoding
        self.crf = CRF(num_tags, batch_first=True)
        
    def _get_emissions(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extracts sequence emission scores from word embeddings.
        Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_tags)
        """
        lstm_out, _ = self.bilstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions
        
    def forward(self, embeddings: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the negative log-likelihood loss of the target tag sequence.
        
        Args:
            embeddings (torch.Tensor): Padded input embeddings of shape (batch_size, seq_len, embedding_dim).
            tags (torch.Tensor): Ground truth tag indices of shape (batch_size, seq_len).
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len) where True indicates non-pad tokens.
            
        Returns:
            torch.Tensor: Negative log-likelihood scalar loss.
        """
        emissions = self._get_emissions(embeddings)
        # We negate the log-likelihood output of pytorch-crf to make it a loss to minimize
        return -self.crf(emissions, tags, mask=mask, reduction='token_mean')
        
    def decode(self, embeddings: torch.Tensor, mask: torch.Tensor) -> list:
        """
        Performs Viterbi decoding to find the highest-scoring sequence of tags.
        
        Args:
            embeddings (torch.Tensor): Padded input embeddings of shape (batch_size, seq_len, embedding_dim).
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len).
            
        Returns:
            list of list of int: The predicted optimal tag sequence per batch instance (variable length list).
        """
        emissions = self._get_emissions(embeddings)
        return self.crf.decode(emissions, mask=mask)
