import torch
import torch.nn as nn
import numpy as np

try:
    from gensim.models import FastText as GensimFastText # type: ignore
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

class FastTextEmbedding(nn.Module):
    """
    FastText word embeddings. Trains a FastText model on the tokenized corpus
    using gensim and initializes a PyTorch nn.Embedding layer with the trained weights.
    FastText leverages subword character n-grams to represent rare or out-of-vocabulary terms.
    
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
                    
                    # Train a FastText model on tokenized sentences
                    model = GensimFastText(
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
                    print(f"  [FastText] Trained FastText on {len(sentences)} sentences. Vocab size: {len(self.vocab)}")
                    return
                except Exception as e:
                    print(f"  [FastText Warning] Training failed: {e}. Falling back to standard embedding.")
            
            # Fallback: either gensim is missing, or training failed
            # Build vocabulary from sentences
            for sent in sentences:
                for token in sent:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
            
            self.embedding = nn.Embedding(len(self.vocab), dim, padding_idx=0)
            if not HAS_GENSIM:
                print(f"  [FastText Warning] gensim is not installed. Initialized a standard trainable PyTorch nn.Embedding with {len(self.vocab)} vocab words.")
            else:
                print(f"  [FastText Warning] Initialized standard trainable PyTorch nn.Embedding with {len(self.vocab)} vocab words.")
        else:
            self.embedding = nn.Embedding(2, dim, padding_idx=0)
            
    def get_dim(self) -> int:
        return self.dim
        
    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_indices)
        
    def text_to_indices(self, tokens_list: list) -> list:
        return [self.vocab.get(token, 1) for token in tokens_list]
