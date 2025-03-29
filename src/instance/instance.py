from src.token.token import Token
from src.vocab.vocab import Vocabulary
from src.vocab.merge import MergeRule
from collections import Counter
from src.token.tokenizer import tokenize_by_merge_rules

# import multiprocessing as mp
# from multiprocessing import Pool
import time
import os

def build_instance(instance_tuple: tuple[str, int]):
    instance_word, instance_count = instance_tuple
    return Instance(instance_word, instance_count)

class Instance:
    def __init__(self, instance_word: str, instance_count: int):
        self.word = instance_word
        self.tokens = []
        
        self.instance_count = instance_count
        
        # initialize token
        for i, subword in enumerate(instance_word):
            subword_token = Token(subword, i != 0)
            self.tokens.append(subword_token)

        ## 맨 처음 부터 초기화하면서 본인이 만들 수 있는 모든 merge rule 후보를 생성함
        self.subword_candidates_pool = set() ### 역인덱싱용
        self.init_subwords_candidates()

        # 현재 토큰 기준 만들 수 있는 bigram 쌍 카운터
        self.token_bigram_merge_rules = []
        self.token_bigram_merge_rules_counter = Counter()
        self.update_token_bigram_merge_rules() ### 현재 토큰 기반으로 생성 가능한 bigram 쌍을 검색하고 카운터 업데이트
    
    def init_subwords_candidates(self):
        self.subword_candidates_pool = set(self.tokens)
        word_len = len(self.word)

        for subword_len in range(2, word_len + 1):
            for start_idx in range(word_len - subword_len + 1):
                substring = self.word[start_idx:start_idx+subword_len]
                candidate_token = Token(substring, is_sub=(start_idx != 0))
                self.subword_candidates_pool.add(candidate_token)

    def update_token_bigram_merge_rules(self):
        self.token_bigram_merge_rules_counter = Counter()

        if len(self.tokens) <= 1:
            return

        for i in range(len(self.tokens)-1):
            token_1 = self.tokens[i]
            token_2 = self.tokens[i+1]

            merge_rule = MergeRule(token_1, token_2)
            self.token_bigram_merge_rules.append(merge_rule) 

            if merge_rule not in self.token_bigram_merge_rules_counter:
                self.token_bigram_merge_rules_counter[merge_rule] = 0

            self.token_bigram_merge_rules_counter[merge_rule] += 1 * self.instance_count
    
    def get_token_list(self):
        return self.tokens

    def get_token_bigram_merge_rules_counter(self):
        return self.token_bigram_merge_rules_counter

    def tokenize(self, vocab: Vocabulary, verbose: bool = False):
        if verbose:
            candidate_tokens = self.subword_candidates_pool
            print(f"candidate_tokens: {[str(token) for token in candidate_tokens]}", end="\n")

            merge_rules_counter = vocab.get_merge_rules_counter(candidate_tokens)
            print(f"merge_rules_counter: {Counter({str(key): merge_rules_counter[key] for key in merge_rules_counter})}", end="\n")
            self.tokens = tokenize_by_merge_rules(self.word, merge_rules_counter, verbose)

            self.update_token_bigram_merge_rules()
        else:
            candidate_tokens = self.subword_candidates_pool

            merge_rules_counter = vocab.get_merge_rules_counter(candidate_tokens)
            self.tokens = tokenize_by_merge_rules(self.word, merge_rules_counter, verbose)

            self.update_token_bigram_merge_rules()

class InstanceManager:
    def __init__(self):
        self.instance_word_to_instance = {}
        self.subword_to_instance = {} ### 역인덱싱용
        
        ## 전체 merge rule 카운터 (이 merge rule은 merge rule 후보군 입니다.)
        self.token_bigram_merge_rules_counter = Counter()

        # instance word to Boolean
        self.tokenize_available_instances = {}
    
    def is_tokenize_available(self):
        return any(self.tokenize_available_instances.values())

    def update_instances(self, subword: Token, vocab: Vocabulary, verbose: bool = False):
        if verbose:
            for instance in self.subword_to_instance[subword]:
                interested_tokens = instance.token_bigram_merge_rules_counter.keys()
                interested_tokens_counter = Counter({str(key): self.token_bigram_merge_rules_counter[key] for key in interested_tokens if key in self.token_bigram_merge_rules_counter})
                
                print("--------------------------------")
                print(f"[before] {instance.word}({subword}):", interested_tokens_counter)
                self.token_bigram_merge_rules_counter -= instance.token_bigram_merge_rules_counter

                interested_tokens_counter2 = Counter({str(key): instance.token_bigram_merge_rules_counter[key] for key in instance.token_bigram_merge_rules_counter}) 
                print("--------------------------------")
                print(f"{instance.word} 재 토큰화 전 사용가능한 토큰 조합 카운터 :", interested_tokens_counter2)

                interested_tokens_counter = Counter({str(key): self.token_bigram_merge_rules_counter[key] for key in interested_tokens if key in self.token_bigram_merge_rules_counter})
                print(f"[after substract] {instance.word}:", interested_tokens_counter)

                instance.tokenize(vocab, verbose)
                print([str(token) for token in instance.tokens])
                interested_tokens_counter2 = Counter({str(key): instance.token_bigram_merge_rules_counter[key] for key in instance.token_bigram_merge_rules_counter})
                print(f"{instance.word} 재 토큰화 후 사용가능한 토큰 조합 카운터 :", interested_tokens_counter2)

                print("--------------------------------")
                self.token_bigram_merge_rules_counter.update(instance.token_bigram_merge_rules_counter)
                interested_tokens_counter = Counter({str(key): self.token_bigram_merge_rules_counter[key] for key in interested_tokens if key in self.token_bigram_merge_rules_counter})
                print(f"[after plus] {instance.word}:", interested_tokens_counter)
                print("--------------------------------")
                self.tokenize_available_instances[instance.word] = len(instance.tokens) > 1
        else:
            for instance in self.subword_to_instance[subword]:
                self.token_bigram_merge_rules_counter -= instance.token_bigram_merge_rules_counter
                
                instance.tokenize(vocab)
                self.token_bigram_merge_rules_counter.update(instance.token_bigram_merge_rules_counter)
                self.tokenize_available_instances[instance.word] = len(instance.tokens) > 1

    def build_instances(self, instance_words: list[str], is_mp_needed: bool = False, mode: str = "train"):
        if mode == "train":
            instance_words_counter = Counter(instance_words)
            total = len(instance_words_counter.keys())
            print(f"total instance 개수: {sum(instance_words_counter.values())}")
            start_time = time.time()
            if is_mp_needed:
                pass
                # num_proc = max(mp.cpu_count() - 1, 1)
                
                # print(f"Using {num_proc} processes for building {total} instances.")
                
                # try:
                #     # Pool을 사용하여 병렬 처리
                #     with Pool(processes=num_proc) as pool:
                #         results = pool.map(build_instance, instance_words_counter.items())
                    
                #     print(f"Built {len(results)} instances.")
                    
                #     # 결과 처리
                #     print("Processing results... (멀티 프로세싱 오버헤드 처리 단계)")
                #     processed_count = 0
                #     for instance in results:
                #         self.instance_word_to_instance[instance.word] = instance
                #         self.tokenize_available_instances[instance.word] = len(instance.tokens) > 1

                #         for subword in instance.token_candidates_pool:
                #             if subword not in self.subword_to_instance:
                #                 self.subword_to_instance[subword] = []
                #             self.subword_to_instance[subword].append(instance)

                #         self.token_bigram_merge_rules_counter.update(instance.token_bigram_merge_rules_counter)
                #         for merge_rule in instance.token_bigram_merge_rules:
                #             if merge_rule not in self.token_bigram_merge_rules_to_instance:
                #                 self.token_bigram_merge_rules_to_instance[merge_rule] = []
                #             self.token_bigram_merge_rules_to_instance[merge_rule].append(instance)
                #         processed_count += 1
                #         print(f"\rProcessing results: {processed_count}/{len(results)}", end="", flush=True)
                #     print("\n")
                    
                #     end_time = time.time()
                #     print(f"Multiprocessing completed in {end_time - start_time:.2f} seconds")
                
                # except KeyboardInterrupt:
                #     print("\nReceived keyboard interrupt. Terminating processes...")
                #     print("Processes terminated.")
                #     raise  # KeyboardInterrupt를 다시 발생시켜 프로그램 종료
            else:
                print(f"Using single process for building {total} instances.")
                
                try:
                    # 단일 프로세스로 처리
                    processed = 0
                    
                    for instance_word, instance_count in instance_words_counter.items():
                        instance = Instance(instance_word, instance_count)
                        self.instance_word_to_instance[instance_word] = instance
                        self.tokenize_available_instances[instance.word] = len(instance.tokens) > 1

                        for subword in instance.subword_candidates_pool:
                            if subword not in self.subword_to_instance:
                                self.subword_to_instance[subword] = []
                            self.subword_to_instance[subword].append(instance)
                        self.token_bigram_merge_rules_counter.update(instance.token_bigram_merge_rules_counter)
                        
                        # 진행 상황 출력
                        processed += 1
                        if processed % 10000 == 0:
                            current_time = time.time()
                            elapsed = current_time - start_time
                            print("\033[2K", end="\r")  # 현재 라인 지우기
                            print(f"Progress: {processed}/{total} instances built. Elapsed: {elapsed:.2f} seconds", end="\r")
                    
                    print("\n")  # 마지막 출력 후 줄바꿈
                    end_time = time.time()
                    print(f"Single processing completed in {end_time - start_time:.2f} seconds")
                
                except KeyboardInterrupt:
                    print("\nReceived keyboard interrupt. Terminating process...")
                    print("Process terminated.")
                    raise  # KeyboardInterrupt를 다시 발생시켜 프로그램 종료
        
        elif mode == "infer":
            # 디버그용 코드
            # print(f"instance 개수: {len(instance_words)}", end = "\n")
            for instance_word in instance_words:
                instance = Instance(instance_word, 1)
                self.instance_word_to_instance[instance_word] = instance

    def update_vocab(self, vocab: Vocabulary):
        for instance in self.instance_word_to_instance.values():
            for token in instance.tokens:
                vocab.add_token(token)

if __name__ == "__main__":
    ## test 1
    # test_word = "huggywoggy"
    # instance = Instance(test_word)
    # print(f"Merge rule candidates for the word '{test_word}':")
    
    # # 후보들을 출력합니다. MergeRule 클래스에 __str__이나 __repr__이 정의되어 있다고 가정합니다.
    # for rule in sorted(list(instance.token_merge_rules_candidates)):
    #     print(rule)

    ## test 2
    from src.util.util import load_corpus
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    corpus_path = os.path.join(current_dir, "..", "..", "data", "train", "pg100.txt")
    corpus = load_corpus(corpus_path)
    
    from src.token.tokenizer import pre_tokenize
    tokenized_instances = pre_tokenize(corpus)
    print(f"pre tokenized instance 개수: {len(tokenized_instances)}")
    print(f"set 적용 후 instance 개수: {len(set(tokenized_instances))}")

    instance_manager = InstanceManager()
    # 멀티 프로세싱 사용
    # instance_manager.build_instances(tokenized_instances, is_mp_needed=True)
    # 멀티 프로세싱 사용 안함 
    ### 비교실험 해봤는데 70000개 기준 멀티 프로세싱 사용 안할때 더 빠름
    instance_manager.build_instances(tokenized_instances, is_mp_needed=False)

    instance = instance_manager.instance_word_to_instance["the"]

    print(instance.word)
    print(",".join([str(token) for token in instance.token_candidates_pool]))
    print("--------------------------------")
    for merge_rule, count in instance.token_bigram_merge_rules_counter.items():
        print(f"{str(merge_rule)}: {count}")