import re
import pprint

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

def split_text(text: str) -> list[str]:
    ret = re.split(r'\n', text)
    while "" in ret:
        ret.remove("")
    return ret

def get_indent(data: dict) -> int:
    pretty_str = pprint.pformat(data)
    indent = 0

    for line in pretty_str.split('\n'):
        _indent = len(line) - len(line.lstrip())
        if _indent > indent:
            indent = _indent

    return indent
