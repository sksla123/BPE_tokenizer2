import re

def load_corpus(corpus_path: str) -> str:
    '''
    코퍼스 파일을 로딩합니다.
    corpus_path (str): 코퍼스 파일 경로

    return (str): 코퍼스 파일 내용
    '''
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
        # BOM 제거
        corpus = corpus.lstrip("\ufeff")
    
    return corpus