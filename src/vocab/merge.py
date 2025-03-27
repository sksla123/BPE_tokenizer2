from src.token.token import Token

class MergeRule(Token):
    def __init__(self, token1: Token, token2: Token):
        super().__init__(token1.token_string + token2.token_string, token1.is_sub)
        
        self.token1 = token1
        self.token2 = token2

    def __str__(self):
        return f"{str(self.token1)} + {str(self.token2)} -> {str(self.token_string)}"
    
    def __hash__(self):
        return hash((self.token1, self.token2))
    
    def __eq__(self, other):
        return self.token1 == other.token1 and self.token2 == other.token2

    def __lt__(self, other):
        return self.token_string < other.token_string