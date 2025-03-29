import argparse
from src.bpe.bpe import BPE

parser = argparse.ArgumentParser(prog="BPE Tokenizer By Merge Rule (Priority)")

## 학습 모드와 추론 모드는 동시에 실행 불가능하게 막아놓음
group = parser.add_mutually_exclusive_group()
## 학습 모드에 사용될 관련 args
group.add_argument('--train', type=str, help='학습에 사용할 코퍼스 파일 위치') # infer와 베타적 그룹
parser.add_argument('--max_vocab', type=int, help='vocab 최대 크기')
parser.add_argument('--vocab', type=str, help='학습 결과를 저장할 vocab 파일 위치')

## 추론 모드에 사용될 관련 args
group.add_argument('--infer', type=str, help='추론에 사용할 저장된 vocab 파일 위치') # train 베타적 그룹
parser.add_argument('--input', type=str, help='추론할 입력 파일(텍스트)')
parser.add_argument('--output', type=str, help='추론된 결과를 저장할 파일 위치')

args = parser.parse_args()

def main():
    if args.train:
        mode = "train"

        config_data = {
            "train_corpus_path": args.train,
            "max_vocab": args.max_vocab,
            "vocab_output_path": args.vocab
        }
    elif args.infer:
        mode = "infer"

        config_data = {
            "infer_vocab_path": args.infer,
            "input_data_path": args.input,
            "tokenized_result_path": args.output
        }

    if mode == "train":
        print("훈련 모드로 프로그램이 동작합니다.")

        bpe = BPE(config_data)
        bpe.train()
        bpe.save_vocab()
    elif mode == "infer":
        print("추론 모드로 프로그램이 동작합니다.")

        bpe = BPE(config_data)
        bpe.load_vocab()
        bpe.infer()

if __name__ == "__main__":
    main()