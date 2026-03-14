import re
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer

nltk.download('stopwords')
stopwords_ru = stopwords.words('russian')
morph = MorphAnalyzer()


def lemmatize(
        doc,
        patterns: str = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if len(token) > 2 and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            
            tokens.append(token)
    
    return tokens