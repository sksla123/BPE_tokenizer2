import re

from src.vocab.vocab import Vocabulary
from queue import PriorityQueue
from src.token.token import Token
from collections import Counter

from src.vocab.merge import MergeRule


def pre_tokenize(corpus: str, method: str = "whitespace"):
    '''
    corpus (str): pre-tokenize 대상 코퍼스
    method (str): pre-tokenize 방법 [whitespace] # 현재는 whitespace만 지원(추후 개발)

    return (list): pre-tokenize 결과(tokenized-instances)
    ''' 

    if method == "whitespace":
        ret = re.split(r'\s+', corpus) ## whitespace 기준으로 분리
        
        # 비어있는 문자 제거
        if "" in ret:
            ret.remove("")
        
        return ret
    else:
        raise ValueError(f"지원하지 않는 pre-tokenize 방법입니다. {method}\n 지원하는 메소드 목록: [whitespace]")

def tokenize_by_merge_rules(word: str, merge_rules_counter: Counter, verbose: bool = False):
    tokens = []
    
    if verbose:
        for i, char in enumerate(word):
            token = Token(char, is_sub=(i != 0))
            tokens.append(token)

        pq = PriorityQueue()

        # 우선순위: (빈도, 토큰 길이, 토큰 문자열)
        # 빈도는 음수로 변환하여 높은 순서로 정렬
        # 토큰 길이는 양수로 사용하여 짧은 순서로 정렬
        for merge_rule, count in merge_rules_counter.items():
            pq.put((-count, len(merge_rule.token_string), merge_rule.token_string, merge_rule))

        print("병합 규칙 목록:", {str(merge_rule): count for merge_rule, count in merge_rules_counter.items()})
        print("토큰 목록:", [str(token) for token in tokens])

        # 병합 규칙 적용
        while not pq.empty():
            _, _, _, merge_rule = pq.get()
            print("병합 규칙:", merge_rule)

            token1, token2 = merge_rule.token1, merge_rule.token2
            
            # 연속된 토큰 쌍을 찾아 병합
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == token1 and tokens[i + 1] == token2:
                    # 토큰 병합
                    merged_token = Token(token1.token_string + token2.token_string, tokens[i].is_sub)
                    tokens[i:i+2] = [merged_token]
                else:
                    i += 1
            
            print("토큰 목록:", [str(token) for token in tokens])
        
        return tokens
    else:
        for i, char in enumerate(word):
            token = Token(char, is_sub=(i != 0))
            tokens.append(token)

        pq = PriorityQueue()

        # 우선순위: (빈도, 토큰 길이, 토큰 문자열)
        for merge_rule, count in merge_rules_counter.items():
            pq.put((-count, len(merge_rule.token_string), merge_rule.token_string, merge_rule))

        # 병합 규칙 적용
        while not pq.empty():
            _, _, _, merge_rule = pq.get()
            token1, token2 = merge_rule.token1, merge_rule.token2
            
            # 연속된 토큰 쌍을 찾아 병합
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == token1 and tokens[i + 1] == token2:
                    # 토큰 병합
                    merged_token = Token(token1.token_string + token2.token_string, tokens[i].is_sub)
                    tokens[i:i+2] = [merged_token]
                else:
                    i += 1
        
        return tokens

if __name__ == "__main__":
    # 테스트 데이터 준비
    word = "The"
    
    # 병합 규칙 생성
    t_token = Token("T", is_sub=False)
    h_token = Token("h", is_sub=True)
    e_token = Token("e", is_sub=True)
    he_token = Token("he", is_sub=True)
    
    merge_rules_counter = Counter()
    merge_rules_counter[MergeRule(t_token, h_token)] = 2000
    merge_rules_counter[MergeRule(h_token, e_token)] = 2500
    merge_rules_counter[MergeRule(t_token, he_token)] = 100
    
    # 토큰화 실행
    tokens = tokenize_by_merge_rules(word, merge_rules_counter)
    
    # 결과 출력
    print(f"입력 단어: {word}")
    print(f"병합 규칙:")
    for rule, count in merge_rules_counter.items():
        print(f"  {rule}: {count}")
    print(f"\n목표 토큰화 결과: ['The']")
    print(f"토큰화 결과: {[str(t) for t in tokens]}")
    