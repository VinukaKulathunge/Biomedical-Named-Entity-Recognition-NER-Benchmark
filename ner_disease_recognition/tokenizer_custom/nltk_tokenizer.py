import nltk
from nltk.tokenize import TreebankWordTokenizer

class NLTKTokenizer:
    """
    An NLTK-based tokenizer that uses the standard TreebankWordTokenizer
    to split text and retrieve exact character boundaries (spans).
    """
    def __init__(self):
        # TreebankWordTokenizer is rule-based and does not require downloading model binaries.
        self.tokenizer = TreebankWordTokenizer()
        
    def tokenize(self, text: str):
        """
        Tokenizes the input text using Treebank rules.
        
        Args:
            text (str): The raw text to tokenize.
            
        Returns:
            tuple: (tokens, spans) where:
                - tokens is a list of token strings.
                - spans is a list of (start_char, end_char) boundaries.
        """
        try:
            spans = list(self.tokenizer.span_tokenize(text))
            tokens = [text[start:end] for start, end in spans]
            return tokens, spans
        except Exception as e:
            # Fallback to simple regex tokenizer if nltk tokenizer fails
            import re
            tokens = []
            spans = []
            # Match word characters or punctuation
            for match in re.finditer(r'\w+|[^\w\s]', text):
                tokens.append(match.group())
                spans.append(match.span())
            return tokens, spans
