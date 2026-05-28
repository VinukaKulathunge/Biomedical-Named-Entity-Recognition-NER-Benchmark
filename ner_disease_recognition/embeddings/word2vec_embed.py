import torch
import torch.nn as nn
import numpy as np

try:
    from gensim.models import Word2Vec as GensimWord2Vec # type: ignore
    import logging
    # Disable gensim verbose logs unless debugging
    logging.getLogger("gensim").setLevel(logging.WARNING)
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

class Word2VecEmbedding(nn.Module):
    """
    Word2Vec word embeddings. Trains a Word2Vec model on the tokenized corpus
    using gensim and initializes a PyTorch nn.Embedding layer with the trained weights.
    
    If gensim is missing, gracefully falls back to a standard PyTorch trainable embedding layer.
    """
    def __init__(self, sentences=None, dim: int = 100, min_count: int = 1, window: int = 5):
        super().__init__()
        self.dim = dim
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        
        if sentences is not None and len(sentences) > 0:
            if HAS_GENSIM:
                try:
                    # Default initialization weights
                    weights = [np.zeros(dim), np.random.normal(scale=0.1, size=dim)] # PAD, UNK
                    
                    # Train Word2Vec model on tokenized sentences
                    model = GensimWord2Vec(
                        sentences, 
                        vector_size=dim, 
                        min_count=min_count, 
                        window=window, 
                        workers=4,
                        epochs=10
                    )
                    
                    # Populating vocabulary and weights
                    for word in model.wv.index_to_key:
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab)
                            weights.append(model.wv[word])
                            
                    weights = np.stack(weights)
                    self.embedding = nn.Embedding.from_pretrained(
                        torch.FloatTensor(weights), 
                        freeze=False, 
                        padding_idx=0
                    )
                    print(f"  [Word2Vec] Trained Word2Vec on {len(sentences)} sentences. Vocab size: {len(self.vocab)}")
                    return
                except Exception as e:
                    print(f"  [Word2Vec Warning] Training failed: {e}. Falling back to standard embedding.")
            
            # Fallback: either gensim is missing, or training failed
            # Build vocabulary from sentences
            for sent in sentences:
                for token in sent:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
            
            self.embedding = nn.Embedding(len(self.vocab), dim, padding_idx=0)
            if not HAS_GENSIM:
                print(f"  [Word2Vec Warning] gensim is not installed. Initialized a standard trainable PyTorch nn.Embedding with {len(self.vocab)} vocab words.")
            else:
                print(f"  [Word2Vec Warning] Initialized standard trainable PyTorch nn.Embedding with {len(self.vocab)} vocab words.")
        else:
            self.embedding = nn.Embedding(2, dim, padding_idx=0)
            
    def get_dim(self) -> int:
        return self.dim
        
    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_indices (torch.Tensor): Integer indices of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Dense word vectors of shape (batch_size, seq_len, dim)
        """
        return self.embedding(token_indices)
        
    def text_to_indices(self, tokens_list: list) -> list:
        """
        Maps a list of string tokens to vocabulary indices.
        """
        return [self.vocab.get(token, 1) for token in tokens_list]
