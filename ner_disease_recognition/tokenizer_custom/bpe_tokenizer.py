from transformers import BertTokenizer

class BPETokenizer:
    """
    A subword tokenizer that wraps Hugging Face's BertTokenizer.
    It splits words into BPE/WordPiece subwords and extracts exact character-level
    offset mappings. Supports both Fast (Rust) and standard (pure-Python) implementations.
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        try:
            from transformers import BertTokenizerFast # type: ignore
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
            self.is_fast = True
        except (ImportError, AttributeError):
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.is_fast = False
            print("  [BPETokenizer Warning] BertTokenizerFast is unavailable (missing compiled Rust package).")
            print("  --> Falling back to pure-Python BertTokenizer with manual character-span matching.")
        
    def tokenize(self, text: str):
        """
        Tokenizes the input text using WordPiece subwords.
        
        Args:
            text (str): The raw text to tokenize.
            
        Returns:
            tuple: (tokens, spans) where:
                - tokens is a list of token strings (e.g. ['[CLS]', 'anti', '##hy', ..., '[SEP]']).
                - spans is a list of (start_char, end_char) boundaries.
        """
        if self.is_fast:
            # Run fast tokenizer and request offset mappings
            encoding = self.tokenizer(
                text, 
                return_offsets_mapping=True, 
                add_special_tokens=True
            )
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
            spans = encoding["offset_mapping"]
            return tokens, spans
        else:
            # Fallback for standard BertTokenizer: manual sequential subword matching
            raw_tokens = self.tokenizer.tokenize(text)
            tokens = ["[CLS]"] + raw_tokens + ["[SEP]"]
            
            spans = [(0, 0)] # CLS span
            current_char_idx = 0
            text_lower = text.lower()
            
            for t in raw_tokens:
                # Strip BPE/WordPiece subword prefix '##'
                clean_t = t.replace("##", "")
                clean_t_lower = clean_t.lower()
                
                # Find the next occurrence of this token in the text
                pos = text_lower.find(clean_t_lower, current_char_idx)
                
                if pos != -1:
                    start = pos
                    end = pos + len(clean_t)
                    spans.append((start, end))
                    current_char_idx = end
                else:
                    # Graceful boundary recovery fallback
                    spans.append((current_char_idx, current_char_idx))
                    
            spans.append((0, 0)) # SEP span
            return tokens, spans
