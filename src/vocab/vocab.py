from src.token.token import Token
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.merge_rules = Counter()
        
        # Add all ASCII characters (0-255) as word tokens and sub tokens using list comprehension
        self.word_tokens = [Token(chr(i), is_sub=False) for i in range(256)]
        self.sub_tokens = [Token(chr(i), is_sub=True) for i in range(256)]
    
    def add_token(self, token: Token):
        if token.is_sub:
            if token not in self.sub_tokens:
                self.sub_tokens.append(token)
        else:
            if token not in self.word_tokens:
                self.word_tokens.append(token)
