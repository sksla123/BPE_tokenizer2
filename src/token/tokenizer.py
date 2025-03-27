import re

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
