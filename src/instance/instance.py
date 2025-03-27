from src.token.token import Token
from src.vocab.vocab import Vocabulary
from src.vocab.merge import MergeRule
from collections import Counter

class Instance:
    def __init__(self, instance_word: str):
        self.word = instance_word
        self.tokens = []
        
        # initialize token
        for i in range(len(instance_word)):
            subword = instance_word[i]
            subword_token = Token(subword, i != 0)

        ## 맨 처음 부터 초기화하면서 본인이 만들 수 있는 모든 merge rule 후보를 생성함함
        self.token_merge_rules_candidates = []
        self.init_merge_rules_candidates()

        # 현재 토큰 기준 만들 수 있는 bigram 쌍 카운터
        self.token_bigram_merge_rules = []
        self.token_bigram_merge_rules_counter = Counter()
        self.update_token_bigram_merge_rules() ## 현재 토큰 기반으로 생성 가능한 bigram 쌍을 검색하고 카운터 업데이트
    
    def init_merge_rules_candidates(self):
        available_tokens = set()

    def update_token_bigram_merge_rules(self):
        for i in range(len(self.tokens)-1):
            token_1 = self.tokens[i]
            token_2 = self.tokens[i+1]

            merge_rule = MergeRule(token_1, token_2)
            self.token_bigram_merge_rules.append(merge_rule)

            if merge_rule not in self.token_bigram_merge_rules_counter:
                self.token_bigram_merge_rules_counter[merge_rule] = 0

            self.token_bigram_merge_rules_counter[merge_rule] += 1
        
        return self.token_bigram_merge_rules



class InstanceManager:
    def __init__(self):
        pass

    def build_instances(self, vocab:Vocabulary=None, ):
        pass