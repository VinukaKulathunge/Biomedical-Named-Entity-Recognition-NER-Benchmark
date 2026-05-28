import torch
import torch.nn as nn

class BiLSTM_Softmax(nn.Module):
    """
    BiLSTM-Softmax sequence labeling model.
    Encodes dense word embeddings using a bidirectional LSTM, maps the output
    to tag space, and decodes labels independently per token using argmax over
    the softmax logits. Trained using cross-entropy loss over non-pad tokens.
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
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)
        
        # CrossEntropyLoss. Setting ignore_index=-1 automatically skips loss computations
        # for padded tokens during training.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
    def forward(self, embeddings: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the token-level cross-entropy loss.
        
        Args:
            embeddings (torch.Tensor): Padded input embeddings of shape (batch_size, seq_len, embedding_dim).
            tags (torch.Tensor): Ground truth tag indices of shape (batch_size, seq_len), 
                                 where padded tokens are pre-marked with -1.
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len) where True indicates non-pad tokens.
            
        Returns:
            torch.Tensor: Cross-entropy scalar loss.
        """
        lstm_out, _ = self.bilstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        logits = self.hidden2tag(lstm_out) # (batch_size, seq_len, num_tags)
        
        # Reshape logits to (batch_size * seq_len, num_tags) and tags to (batch_size * seq_len)
        flat_logits = logits.view(-1, self.num_tags)
        flat_tags = tags.view(-1)
        
        loss = self.loss_fn(flat_logits, flat_tags)
        return loss
        
    def decode(self, embeddings: torch.Tensor, mask: torch.Tensor) -> list:
        """
        Predicts the optimal tag sequence by taking token-level argmax.
        
        Args:
            embeddings (torch.Tensor): Padded input embeddings of shape (batch_size, seq_len, embedding_dim).
            mask (torch.Tensor): Boolean mask of shape (batch_size, seq_len).
            
        Returns:
            list of list of int: Predicted tag indices per sequence, ignoring padded tail tokens.
        """
        lstm_out, _ = self.bilstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        logits = self.hidden2tag(lstm_out) # (batch_size, seq_len, num_tags)
        
        # Get independent argmax predictions
        preds = torch.argmax(logits, dim=-1) # (batch_size, seq_len)
        
        # Convert to variable-length sequences of integer lists according to the mask
        decoded = []
        for i in range(preds.size(0)):
            seq_len = mask[i].sum().item()
            decoded.append(preds[i, :seq_len].tolist())
            
        return decoded
