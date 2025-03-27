from src.token.token import Token
from src.vocab.vocab import Vocabulary
from src.vocab.merge import MergeRule
from collections import Counter

import multiprocessing as mp
import time

def build_instance(instance_word:str):
    return Instance(instance_word)

class Instance:
    def __init__(self, instance_word: str):
        self.word = instance_word
        self.tokens = []
        
        # initialize token
        for i, subword in enumerate(instance_word):
            subword_token = Token(subword, i != 0)
            self.tokens.append(subword_token)

        ## 맨 처음 부터 초기화하면서 본인이 만들 수 있는 모든 merge rule 후보를 생성함
        self.token_subwords_candidates = set() ### 역인덱싱용
        self.init_subwords_candidates()

        # 현재 토큰 기준 만들 수 있는 bigram 쌍 카운터
        self.token_bigram_merge_rules = []
        self.token_bigram_merge_rules_counter = Counter()
        self.update_token_bigram_merge_rules() ### 현재 토큰 기반으로 생성 가능한 bigram 쌍을 검색하고 카운터 업데이트
    
    def init_subwords_candidates(self):
        self.token_subwords_candidates = set()
        word_len = len(self.word)
        # 모든 가능한 연속된 substring 후보 생성 (길이 2 이상)
        for subword_len in range(2, word_len + 1):
            for start_idx in range(word_len - subword_len + 1):
                substring = self.word[start_idx:start_idx+subword_len]
                # 시작 인덱스가 0이면 단어의 첫 토큰, 아니면 subword로 취급
                candidate_token = Token(substring, is_sub=(start_idx != 0))
                self.token_subwords_candidates.add(candidate_token)

    def update_token_bigram_merge_rules(self):
        for i in range(len(self.tokens)-1):
            token_1 = self.tokens[i]
            token_2 = self.tokens[i+1]

            merge_rule = MergeRule(token_1, token_2)
            self.token_bigram_merge_rules.append(merge_rule) 

            if merge_rule not in self.token_bigram_merge_rules_counter:
                self.token_bigram_merge_rules_counter[merge_rule] = 0

            self.token_bigram_merge_rules_counter[merge_rule] += 1
    
    def get_token_list(self):
        return self.tokens

    def get_token_bigram_merge_rules_counter(self):
        return self.token_bigram_merge_rules_counter

class InstanceManager:
    def __init__(self, instance_words: list[str]):
        self.instance_word_to_instance = {}
        self.subword_to_instance = {} ### 역인덱싱용
        
        ## 전체 merge rule 카운터
        self.token_bigram_merge_rules_counter = Counter()

    def build_instances(self, instance_words: list[str], is_mp_needed: bool = False):
        if is_mp_needed:
            num_proc = max(mp.cpu_count() - 1, 1)
            total = len(instance_words)
            processed = 0

            print(f"Using {num_proc} processes for building {total} instances.")
            global_start_time = time.time()
            last_print_time = time.time()

            with mp.Pool(processes=num_proc) as pool:
                results = []
                for instance in pool.imap_unordered(build_instance, instance_words):
                    results.append(instance)
                    processed += 1
                    current_time = time.time()
                    # 2초마다 진행 상황 업데이트 (전체 경과 시간 포함)
                    if current_time - last_print_time >= 2:
                        elapsed = current_time - global_start_time
                        print(f"\rProgress: {processed}/{total} instances built. Elapsed: {elapsed:.2f} seconds", end="", flush=True)
                        last_print_time = current_time

            print()
            
            # mp 오버헤드
            for instance in results:
                self.instance_word_to_instance[instance.word] = instance
                for subword in instance.token_subwords_candidates:
                    if subword not in self.subword_to_instance:
                        self.subword_to_instance[subword] = []
                    self.subword_to_instance[subword].append(instance)
                self.token_bigram_merge_rules_counter.update(instance.token_bigram_merge_rules_counter)
        else:
            # 단일 프로세스로 처리
            for instance_word in instance_words:
                instance = Instance(instance_word)
                self.instance_word_to_instance[instance_word] = instance

                for subword in instance.token_subwords_candidates:
                    if subword not in self.subword_to_instance:
                        self.subword_to_instance[subword] = []
                    self.subword_to_instance[subword].append(instance)
                self.token_bigram_merge_rules_counter.update(instance.token_bigram_merge_rules_counter)



if __name__ == "__main__":
    test_word = "huggywoggy"
    instance = Instance(test_word)
    print(f"Merge rule candidates for the word '{test_word}':")
    
    # 후보들을 출력합니다. MergeRule 클래스에 __str__이나 __repr__이 정의되어 있다고 가정합니다.
    for rule in sorted(list(instance.token_merge_rules_candidates)):
        print(rule)