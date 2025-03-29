from src.token.token import Token
from src.vocab.vocab import Vocabulary
from src.vocab.merge import MergeRule
from src.instance.instance import InstanceManager
from src.token.tokenizer import pre_tokenize
from src.util.util import load_corpus, split_text
from collections import Counter
import time

class BPE:
    def __init__(self, config_data: dict):
        self.config_data = config_data
        self.train_corpus_path = config_data.get("train_corpus_path", None)
        self.max_vocab_size = config_data.get("max_vocab", None)
        self.vocab_save_path = config_data.get("vocab_output_path", None)
        self.infer_vocab_path = config_data.get("infer_vocab_path", None)
        self.input_data_path = config_data.get("input_data_path", None)
        self.tokenized_result_path = config_data.get("tokenized_result_path", None)
        
        self.corpus = ""
        self.pre_tokenized_corpus = None
        self.vocab = Vocabulary()

        self.instance_manager = InstanceManager()
    
    def initialize_vocab(self):
        tokens = set()

        for instance in self.instance_manager.instance_word_to_instance.values():
            tokens = tokens | set(instance.tokens)

        for token in tokens:
            self.vocab.add_token(token)

    def train(self):
        """
        BPE 알고리즘을 사용하여 어휘를 학습합니다.
        
        Args:
            corpus (str): 학습할 텍스트 코퍼스
            is_mp_needed (bool): 멀티프로세싱 사용 여부
        """
        print("Starting BPE training...")
        start_time = time.time()
        
        self.corpus = load_corpus(self.train_corpus_path)
        self.pre_tokenized_corpus = pre_tokenize(self.corpus)

        self.instance_manager.build_instances(self.pre_tokenized_corpus, is_mp_needed=False)
        
        print("vocab 초기화 중...")
        self.initialize_vocab()
        print("vocab 초기화 완료")
        current_vocab_size = self.vocab.get_vocab_size()
        print(f"현재 vocab size: {current_vocab_size}")
        
        print("BPE 학습 중...")
        train_count = 0
        start_time = time.time()
        while current_vocab_size < self.max_vocab_size:
            self._train()
            train_count += 1
            current_vocab_size = self.vocab.get_vocab_size()

            if train_count % 10 == 1:
                elapsed_time = time.time() - start_time
                print(f"\r현재 학습 횟수: {train_count} | 현재 vocab size: {current_vocab_size} | 경과 시간: {elapsed_time:.2f}초", end="")

            # 디버깅 코드
            # for word, is_available in self.instance_manager.tokenize_available_instances.items():
            #     if is_available:
            #         print("token list[", word, "]:", [str(token) for token in self.instance_manager.instance_word_to_instance[word].tokens])
                
            if not self.instance_manager.is_tokenize_available():
                break
        
        elapsed_time = time.time() - start_time
        print(f"\r현재 학습 횟수: {train_count} | 현재 vocab size: {current_vocab_size} | 경과 시간: {elapsed_time:.2f}초")

        end_time = time.time()
        print(f"BPE training completed in {end_time - start_time:.2f} seconds")
        print(f"Final vocabulary size: {len(self.vocab.word_tokens) + len(self.vocab.sub_tokens)}")
    
    def _train(self):
        merge_rule, count = self.instance_manager.token_bigram_merge_rules_counter.most_common(1)[0]
        
        f = self.vocab.add_merge_rule(merge_rule, count)
        if not f:
            print("\n\n이미 존재하는 merge rule입니다.:", merge_rule, count)
            self.instance_manager.update_instances(merge_rule.get_merged_token(), self.vocab, verbose=True)
            os._exit(0)

        
        merged_token = merge_rule.get_merged_token()

        self.instance_manager.update_instances(merged_token, self.vocab)

    def save_vocab(self):
        self.vocab.save_vocab(self.vocab_save_path)

    def load_vocab(self):
        self.vocab.load_vocab(self.infer_vocab_path)

    def infer(self):
        self.corpus = load_corpus(self.input_data_path)
        
        infer_sentences = split_text(self.corpus)

        infer_output = ""
        total_count = len(infer_sentences)
        for i, sentence in enumerate(infer_sentences):
            print(f"\rProgress: {i+1}/{total_count}, sentence: {sentence}", end=" ")
            pre_tokenized_sentence = pre_tokenize(sentence)
            self.instance_manager.build_instances(pre_tokenized_sentence, is_mp_needed=False, mode="infer")

            tokens = []
            for pre_token in pre_tokenized_sentence:
                instance = self.instance_manager.instance_word_to_instance[pre_token]
                # 디버그용 코드
                # instance.tokenize(self.vocab, verbose=True)
                instance.tokenize(self.vocab)
                tokens.extend(instance.tokens)
            tokens = [token.token_string if not token.is_sub else "##" + token.token_string for token in tokens]
            
            infer_output += " ".join(tokens) + "\n"
        print("\n")

        print("토크나이즈 결과 저장 중..., to:", self.tokenized_result_path)
        with open(self.tokenized_result_path, "w", encoding="utf-8") as f:
            f.write(infer_output)
        print("토크나이즈 결과 저장 완료")

if __name__ == "__main__":
    # 테스트 코드
    from src.util.util import load_corpus
    import os
    
    # 코퍼스 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(current_dir, "..", "..", "data", "train", "pg100.txt")
    vocab_save_path = os.path.join(current_dir, "..", "..", "data", "vocab", "vocab.json")
    input_data_path = os.path.join(current_dir, "..", "..", "data", "infer", "input", "input.txt")
    output_data_path = os.path.join(current_dir, "..", "..", "data", "infer", "output", "output.txt")
    
    # BPE 모델 학습
    # config_data = {
    #     "train_corpus_path": corpus_path,
    #     "vocab_output_path": vocab_save_path,
    #     "max_vocab": 30000
    # }
    # bpe = BPE(config_data)
    # bpe.train()
    # bpe.vocab.save_vocab(bpe.vocab_save_path)


    # print("vocab size:", bpe.vocab.get_vocab_size())
    # print("vocab word size:", len(bpe.vocab.word_tokens))
    # print("vocab sub size:", len(bpe.vocab.sub_tokens))
    # print("vocab merge rules size:", len(bpe.vocab.merge_rules))

    config_data = {
        "input_data_path": input_data_path,
        "infer_vocab_path": vocab_save_path,
        "tokenized_result_path": output_data_path
    }

    bpe = BPE(config_data)
    bpe.load_vocab()

    print("vocab size:", bpe.vocab.get_vocab_size())
    print("vocab word size:", len(bpe.vocab.word_tokens))
    print("vocab sub size:", len(bpe.vocab.sub_tokens))
    print("vocab merge rules size:", len(bpe.vocab.merge_rules))
    # os._exit(0)

    bpe.infer()
    