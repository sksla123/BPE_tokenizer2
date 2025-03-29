from src.token.token import Token
from src.vocab.vocab import Vocabulary
from src.vocab.merge import MergeRule
from src.instance.instance import InstanceManager
from src.token.tokenizer import pre_tokenize
from collections import Counter
import time

class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = Vocabulary()
        self.instance_manager = None

    def init_instance_manager(self, tokenized_instances: list[list[Token]]):
    
    def train(self, corpus: str, is_mp_needed: bool = False):
        """
        BPE 알고리즘을 사용하여 어휘를 학습합니다.
        
        Args:
            corpus (str): 학습할 텍스트 코퍼스
            is_mp_needed (bool): 멀티프로세싱 사용 여부
        """
        print("Starting BPE training...")
        start_time = time.time()
        
        # 1. 코퍼스를 단어 단위로 분리
        print("Pre-tokenizing corpus...")
        tokenized_instances = pre_tokenize(corpus)
        print(f"Pre-tokenized {len(tokenized_instances)} instances")
        
        # 2. 인스턴스 매니저 초기화 및 인스턴스 생성
        print("Building instances...")
        self.instance_manager = InstanceManager(tokenized_instances)
        self.instance_manager.build_instances(tokenized_instances, is_mp_needed)
        
        # 3. BPE 병합 규칙 학습
        print("Learning merge rules...")
        current_vocab_size = len(self.vocab.word_tokens) + len(self.vocab.sub_tokens)
        num_merges = self.vocab_size - current_vocab_size
        
        for i in range(num_merges):
            # 가장 빈도가 높은 bigram 쌍 찾기
            if not self.instance_manager.token_bigram_merge_rules_counter:
                print(f"No more merge rules available. Stopping at {i} merges.")
                break
                
            best_merge_rule = max(
                self.instance_manager.token_bigram_merge_rules_counter.items(),
                key=lambda x: x[1]
            )[0]
            
            # 새로운 토큰 생성 및 어휘에 추가
            new_token = Token(best_merge_rule.token_string, best_merge_rule.is_sub)
            self.vocab.add_token(new_token)
            
            # 인스턴스 업데이트
            self._apply_merge_rule(best_merge_rule)
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{num_merges} merges")
        
        end_time = time.time()
        print(f"BPE training completed in {end_time - start_time:.2f} seconds")
        print(f"Final vocabulary size: {len(self.vocab.word_tokens) + len(self.vocab.sub_tokens)}")
    
    def _apply_merge_rule(self, merge_rule: MergeRule):
        """
        병합 규칙을 적용하여 인스턴스들을 업데이트합니다.
        
        Args:
            merge_rule (MergeRule): 적용할 병합 규칙
        """
        # 병합 규칙이 적용된 인스턴스들을 찾습니다
        affected_instances = []
        for instance in self.instance_manager.instance_word_to_instance.values():
            if merge_rule.token1 in instance.tokens and merge_rule.token2 in instance.tokens:
                affected_instances.append(instance)
        
        # 각 인스턴스에 대해 병합 규칙을 적용합니다
        for instance in affected_instances:
            # 토큰 리스트에서 병합할 토큰들을 찾아 새로운 토큰으로 교체
            new_tokens = []
            i = 0
            while i < len(instance.tokens):
                if i < len(instance.tokens) - 1 and instance.tokens[i] == merge_rule.token1 and instance.tokens[i + 1] == merge_rule.token2:
                    new_token = Token(merge_rule.token_string, merge_rule.is_sub)
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(instance.tokens[i])
                    i += 1
            
            # 인스턴스 업데이트
            instance.tokens = new_tokens
            instance.update_token_bigram_merge_rules()
            
            # 병합 규칙 카운터 업데이트
            self.instance_manager.token_bigram_merge_rules_counter[merge_rule] = 0
    
    def tokenize(self, text: str) -> list[Token]:
        """
        학습된 BPE 모델을 사용하여 텍스트를 토큰화합니다.
        
        Args:
            text (str): 토큰화할 텍스트
            
        Returns:
            list[Token]: 토큰화된 결과
        """
        # 1. 텍스트를 단어 단위로 분리
        words = pre_tokenize(text)
        
        # 2. 각 단어에 대해 BPE 토큰화 적용
        tokens = []
        for word in words:
            word_tokens = [Token(c, i != 0) for i, c in enumerate(word)]
            
            # 가능한 모든 병합 규칙을 적용
            while True:
                # 현재 토큰들 사이의 모든 bigram 쌍을 확인
                best_merge = None
                best_merge_rule = None
                
                for i in range(len(word_tokens) - 1):
                    token1 = word_tokens[i]
                    token2 = word_tokens[i + 1]
                    merge_rule = MergeRule(token1, token2)
                    
                    # 병합 규칙이 어휘에 있는지 확인
                    if merge_rule.token_string in [t.token_string for t in self.vocab.word_tokens + self.vocab.sub_tokens]:
                        if best_merge is None or merge_rule.token_string < best_merge.token_string:
                            best_merge = merge_rule
                            best_merge_rule = (i, i + 1)
                
                if best_merge is None:
                    break
                
                # 가장 좋은 병합 규칙 적용
                i, j = best_merge_rule
                new_token = Token(best_merge.token_string, best_merge.is_sub)
                word_tokens[i:j+1] = [new_token]
            
            tokens.extend(word_tokens)
        
        return tokens

if __name__ == "__main__":
    # 테스트 코드
    from src.util.util import load_corpus
    import os
    
    # 코퍼스 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(current_dir, "..", "..", "data", "pg100.txt")
    corpus = load_corpus(corpus_path)
    
    # BPE 모델 학습
    bpe = BPE(vocab_size=1000)
    bpe.train(corpus, is_mp_needed=False)
    
    # 테스트 텍스트 토큰화
    test_text = "Hello, world!"
    tokens = bpe.tokenize(test_text)
    print(f"Original text: {test_text}")
    print(f"Tokenized: {' '.join(str(t) for t in tokens)}")
