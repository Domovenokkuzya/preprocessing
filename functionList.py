import re

from nltk import word_tokenize, RegexpTokenizer, WordNetLemmatizer, pos_tag

from fieldnames import alphabet_rus
from stopwordsCreation import rus_stop_words


# Первая часть предобработки текста, до токенизации
def pre1(text):
    # lowercasing
    text = text.lower()
    # removing extra whitespaces
    text = " ".join(text.split())
    # join words with -
    ind = 0
    for symb in text:
        if symb == '-' and ind + 3 < len(text) and text[ind + 1] == ' ':
            text = text[:ind] + text[ind + 1:]
            text = text[:ind] + text[ind + 1:]
            ind -= 2
        ind += 1
    # removing urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    return text


# Вторая часть преобразования текста, с токенизацией и после
def pre2(text):
    # tokenization
    text = word_tokenize(text, "russian")
    # removing punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    text = tokenizer.tokenize(' '.join(text))
    # removing russian stopwords and russian litters and numbers
    ind = 0
    for word in text:
        while ind < len(text) and (text[ind] in rus_stop_words or text[ind].isnumeric() or text[ind] in alphabet_rus):
            text.pop(ind)
        ind += 1
    # lemmatization
    result = []
    wordnet = WordNetLemmatizer()
    for token, tag in pos_tag(text):
        pos = tag[0].lower()
        if pos not in ['a', 'r', 'n', 'v']:
            pos = 'n'
        result.append(wordnet.lemmatize(token, pos))
    text = result
    return text


def red(string, ishodn):
    if string in ishodn:
        num = ishodn.find(string)
        ishodn = ishodn[0:num]
    return ishodn


# Функция для подсчета частоты появления токена
def frequent_words(text, lst):
    for token in text:
        lst.append(token)
