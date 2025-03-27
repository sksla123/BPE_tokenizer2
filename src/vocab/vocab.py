from src.token.token import Token
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word_tokens = []
        self.sub_tokens = []
        self.merge_rules = Counter()