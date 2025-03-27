class Token:
    def __init__(self, token_string: str, is_sub: bool):
        self.token_string = token_string
        self.is_sub = is_sub

    def __str__(self):
        return self.token_string if not self.is_sub else f"##{self.token_string}"
    
    def __hash__(self):
        return hash(self.token_string)
    
    def __eq__(self, other):
        return self.token_string == other.token_string and self.is_sub == other.is_sub