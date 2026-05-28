import os
import torch
import torch.nn as nn
import numpy as np

try:
    from gensim.models import Word2Vec as GensimWord2Vec # type: ignore
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

class GloVeEmbedding(nn.Module):
    """
    GloVe word embeddings. Loads pretrained 50d GloVe vectors from a text file.
    If the file is not found, it falls back to training a 50d Word2Vec model on the
    corpus, ensuring the project remains runnable out-of-the-box.
    
    If gensim is missing, gracefully falls back to a standard PyTorch trainable embedding layer.
    """
    def __init__(self, sentences=None, dim: int = 50, glove_path: str = "data/glove.6B.50d.txt"):
        super().__init__()
        self.dim = dim
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        
        # Check alternative locations for convenience
        paths_to_try = [
            glove_path,
            "glove.6B.50d.txt",
            os.path.join(os.path.dirname(__file__), "..", "data", "glove.6B.50d.txt"),
            os.path.join(os.path.dirname(__file__), "glove.6B.50d.txt")
        ]
        
        actual_path = None
        for p in paths_to_try:
            if os.path.exists(p):
                actual_path = p
                break
                
        if actual_path is not None:
            print(f"  [GloVe] Loading GloVe embeddings from {actual_path}...")
            try:
                vectors = []
                vocab_idx = 2
                
                with open(actual_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == dim + 1:
                            word = parts[0]
                            # Handle potential malformed lines
                            try:
                                vec = np.array([float(x) for x in parts[1:]])
                                self.vocab[word] = vocab_idx
                                vectors.append(vec)
                                vocab_idx += 1
                            except ValueError:
                                continue
                                
                weights = [np.zeros(dim), np.random.normal(scale=0.1, size=dim)] # PAD, UNK
                weights.extend(vectors)
                weights = np.stack(weights)
                
                # Freeze pre-trained weights
                self.embedding = nn.Embedding.from_pretrained(
                    torch.FloatTensor(weights), 
                    freeze=True, 
                    padding_idx=0
                )
                print(f"  [GloVe] Successfully loaded {len(self.vocab)} vectors from {actual_path}.")
            except Exception as e:
                print(f"  [GloVe Error] Failed to parse GloVe file: {e}. Falling back to random-initialized embedding.")
                self.embedding = nn.Embedding(2, dim, padding_idx=0)
        else:
            print(f"  [GloVe Warning] glove.6B.50d.txt not found at standard paths.")
            print(f"  --> Falling back to automated corpus vocabulary representation.")
            
            if sentences is not None and len(sentences) > 0:
                if HAS_GENSIM:
                    try:
                        # Train a smaller 50d Word2Vec model as fallback
                        weights = [np.zeros(dim), np.random.normal(scale=0.1, size=dim)]
                        model = GensimWord2Vec(
                            sentences, 
                            vector_size=dim, 
                            min_count=1, 
                            window=5, 
                            workers=4,
                            epochs=10
                        )
                        
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
                        print(f"  [GloVe Fallback] Trained 50d Word2Vec on {len(sentences)} sentences. Vocab size: {len(self.vocab)}")
                        return
                    except Exception as e:
                        print(f"  [GloVe Fallback Warning] Word2Vec training failed: {e}. Falling back to standard embedding.")
                
                # Build vocabulary from sentences as fallback (either gensim is missing, or Word2Vec failed)
                for sent in sentences:
                    for token in sent:
                        if token not in self.vocab:
                            self.vocab[token] = len(self.vocab)
                
                self.embedding = nn.Embedding(len(self.vocab), dim, padding_idx=0)
                if not HAS_GENSIM:
                    print(f"  [GloVe Fallback Warning] gensim is not installed. Initialized a standard trainable PyTorch nn.Embedding with {len(self.vocab)} vocab words.")
                else:
                    print(f"  [GloVe Fallback Warning] Initialized standard trainable PyTorch nn.Embedding with {len(self.vocab)} vocab words.")
            else:
                self.embedding = nn.Embedding(2, dim, padding_idx=0)
                
    def get_dim(self) -> int:
        return self.dim
        
    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_indices)
        
    def text_to_indices(self, tokens_list: list) -> list:
        return [self.vocab.get(token, 1) for token in tokens_list]
