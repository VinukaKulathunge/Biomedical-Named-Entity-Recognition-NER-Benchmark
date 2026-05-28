import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
import numpy as np

class BertEmbedding(nn.Module):
    """
    BERT contextual embeddings. Loads a pre-trained BERT model (default: bert-base-uncased)
    and freezes its weights to act as a high-performance feature extractor.
    
    Contains a robust dynamic alignment algorithm that works on ANY tokenization input
    (Whitespace, NLTK, or BPE) by mapping BERT's internal subword embeddings back to the
    input tokens using exact character-span offsets.
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        
        # Load pre-trained BERT
        print(f"  [BERT] Loading pre-trained {model_name}...")
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
        # Freeze parameters to act as feature extractor
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.dim = self.bert.config.hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(self.device)
        self.bert.eval()
        
        print(f"  [BERT] Frozen feature extractor initialized on {self.device}. Output dim: {self.dim}")

    def get_dim(self) -> int:
        return self.dim
        
    def forward(self, batch_tokens_list: list, max_seq_len: int = None) -> torch.Tensor:
        """
        Extracts contextualized representations for a batch of token lists.
        Handles token alignment internally.
        
        Args:
            batch_tokens_list (list of list of str): Batch of tokenized sentences.
            max_seq_len (int, optional): Truncate or pad to this length.
            
        Returns:
            torch.Tensor: Aligned BERT embeddings of shape (batch_size, seq_len, dim)
        """
        batch_size = len(batch_tokens_list)
        if max_seq_len is None:
            max_seq_len = max(len(s) for s in batch_tokens_list)
            
        # Initialize zero tensor for outputs
        embedded_batch = torch.zeros(
            batch_size, 
            max_seq_len, 
            self.dim, 
            device=self.device
        )
        
        # We process sentences one by one or in a mini-batch to handle dynamic alignment
        with torch.no_grad():
            for i, tokens in enumerate(batch_tokens_list):
                if len(tokens) == 0:
                    continue
                    
                # 1. Reconstruct sentence and track character boundaries of the input tokens
                text = ""
                token_spans = []
                for t_idx, token in enumerate(tokens):
                    start = len(text)
                    text += token
                    end = len(text)
                    token_spans.append((start, end))
                    if t_idx < len(tokens) - 1:
                        text += " "
                        
                # 2. Tokenize with BERT and get subword offset mapping
                encoding = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    return_offsets_mapping=True, 
                    truncation=True, 
                    max_length=512
                )
                
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                subword_spans = encoding["offset_mapping"][0].cpu().numpy() # shape: (bert_seq_len, 2)
                
                # 3. Extract hidden states from BERT
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state[0] # shape: (bert_seq_len, dim)
                
                # 4. Align subword representations back to original tokens
                for t_idx, (orig_start, orig_end) in enumerate(token_spans):
                    if t_idx >= max_seq_len:
                        break
                        
                    # Find BERT subwords that overlap with this token
                    overlapping_vectors = []
                    
                    for sub_idx, (sub_start, sub_end) in enumerate(subword_spans):
                        # Skip special tokens with span (0, 0)
                        if sub_start == 0 and sub_end == 0:
                            continue
                            
                        # Check overlap
                        if not (sub_end <= orig_start or sub_start >= orig_end):
                            overlapping_vectors.append(last_hidden_state[sub_idx])
                            
                    # Average the overlapping vectors to represent the original token
                    if len(overlapping_vectors) > 0:
                        embedded_batch[i, t_idx] = torch.stack(overlapping_vectors).mean(dim=0)
                    else:
                        # Fallback if no overlap (e.g. truncated)
                        embedded_batch[i, t_idx] = torch.zeros(self.dim, device=self.device)
                        
        return embedded_batch
