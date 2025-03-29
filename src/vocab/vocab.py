from src.token.token import Token
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.merge_rules = Counter()
        
        # Add all ASCII characters (0-255) as word tokens and sub tokens using list comprehension
        self.word_tokens = set([Token(chr(i), is_sub=False) for i in range(256)])
        self.sub_tokens = set([Token(chr(i), is_sub=True) for i in range(256)])

        self.tokens = self.word_tokens | self.sub_tokens
        self.vocab_size = len(self.tokens)
    
    def add_token(self, token: Token):
        if token.is_sub:
            if token not in self.sub_tokens:
                self.sub_tokens.add(token)
        else:
            if token not in self.word_tokens:
                self.word_tokens.add(token)
        
        if token not in self.tokens:
            self.tokens.add(token)
        
        self.vocab_size +=1

    def add_merge_rule(self, merge_rule: MergeRule, count: int = 1):
        if merge_rule not in self.merge_rules:
            self.merge_rules[merge_rule] = 0
        self.merge_rules[merge_rule] += count

        token = merge_rule.token
        self.add_token(token)

    def get_vocab_size(self):
        return self.vocab_size

    def get_merge_rules_counter(self, merge_rules: set[MergeRule]):
        return Counter({rule: count for rule, count in self.merge_rules.items() if rule in merge_rules})