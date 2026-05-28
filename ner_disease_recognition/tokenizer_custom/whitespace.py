import re

class WhitespaceTokenizer:
    """
    A simple tokenizer that splits text based on whitespace and returns
    tokens alongside their exact start and end character offsets.
    """
    def __init__(self):
        pass
        
    def tokenize(self, text: str):
        """
        Tokenizes the input text by whitespace.
        
        Args:
            text (str): The raw text to tokenize.
            
        Returns:
            tuple: (tokens, spans) where:
                - tokens is a list of token strings.
                - spans is a list of (start_char, end_char) integer boundaries.
        """
        tokens = []
        spans = []
        
        # Match any sequence of non-whitespace characters
        for match in re.finditer(r'\S+', text):
            tokens.append(match.group())
            spans.append(match.span())
            
        return tokens, spans
