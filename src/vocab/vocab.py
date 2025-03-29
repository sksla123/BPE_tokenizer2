from src.token.token import Token
from collections import Counter
from src.vocab.merge import MergeRule
import json
from src.util.util import get_indent

class Vocabulary:
    def __init__(self):
        self.merge_rules = set()
        self.merge_rules_counter = Counter()
        self.merged_token_to_merge_rule = {}
        
        # Add all ASCII characters (0-255) as word tokens and sub tokens using list comprehension
        self.word_tokens = set([Token(chr(i), is_sub=False) for i in range(256)])
        self.sub_tokens = set([Token(chr(i), is_sub=True) for i in range(256)])

        self.tokens = self.word_tokens | self.sub_tokens
        self.vocab_size = len(self.tokens)
    
    def add_token(self, token: Token):
        if token.is_sub:
            if token not in self.sub_tokens:
                self.sub_tokens.add(token)
                self.vocab_size += 1
        else:
            if token not in self.word_tokens:
                self.word_tokens.add(token)
                self.vocab_size += 1

        if token not in self.tokens:
            self.tokens.add(token)

    def add_merge_rule(self, merge_rule: MergeRule, count: int):
        merged_rule_token = merge_rule.get_merged_token()

        if merged_rule_token not in self.merge_rules_counter.keys():
            self.merge_rules.add(merge_rule)
            self.merge_rules_counter[merged_rule_token] = count
            self.merged_token_to_merge_rule[merged_rule_token] = merge_rule
            self.add_token(merged_rule_token)

            return True
        
        return False

    def get_vocab_size(self):
        return self.vocab_size

    def get_merge_rules_counter(self, merge_rule_tokens: set[Token]):
        counter = Counter()
        for token in merge_rule_tokens:
            if token in self.merge_rules_counter.keys():
                counter[self.merged_token_to_merge_rule[token]] = self.merge_rules_counter[token]

        return counter

    def save_vocab(self, save_path: str):
        print("vocab 저장 중..., to:", save_path)
        with open(save_path, "w") as f:
            dict_data = {
                "word_vocab": [str(token) for token in self.word_tokens],
                "sub_vocab": [str(token) for token in self.sub_tokens],
                "merge_rules": [str(merge_rule) + '\t' + str(self.merge_rules_counter[merge_rule.get_merged_token()]) for merge_rule in self.merge_rules]
            }
            
            indent = get_indent(dict_data)
            
            json.dump(dict_data, f, indent=indent)
        print("vocab 저장 완료")

    def load_vocab(self, load_path: str):
        print("vocab 로드 중..., from:", load_path)
        with open(load_path, "r") as f:
            dict_data = json.load(f)
            self.word_tokens = set([Token(token, is_sub=False) for token in dict_data["word_vocab"]])
            self.sub_tokens = set([Token(token, is_sub=True) for token in dict_data["sub_vocab"]])
            self.tokens = self.word_tokens | self.sub_tokens
            self.vocab_size = len(self.tokens)

            for str_merge_rule in dict_data["merge_rules"]:
                _str_merge_rule = str_merge_rule.split("\t")
                
                if _str_merge_rule[0].startswith("[ sub ]"):
                    token1 = Token(_str_merge_rule[0][7:], is_sub=True)
                else:
                    token1 = Token(_str_merge_rule[0], is_sub=False)

                if _str_merge_rule[1].startswith("[ sub ]"):
                    token2 = Token(_str_merge_rule[1][7:], is_sub=True)
                else:
                    token2 = Token(_str_merge_rule[1], is_sub=False)

                merge_rule = MergeRule(token1, token2)
                
                # 디버그 용 코드
                # print(f"merge_rule: {str(merge_rule)}, count: {int(_str_merge_rule[3])}")
                
                self.add_merge_rule(merge_rule, int(_str_merge_rule[3]))
        
        # 디버그 용 코드
        # import os
        # with open("./vocab_debug.txt", "w", encoding="utf-8") as f:
        #     f.write(f"merge_rules_counter: {Counter({str(key): self.merge_rules_counter[key] for key in self.merge_rules_counter})}")
        # os._exit(0)

        print("vocab 로드 완료")
