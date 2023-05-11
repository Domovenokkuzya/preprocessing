import os
import csv

from nltk import corpus
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

from fieldnames import GRNTI, alphabet_rus

# Путь до папки с таблицами csv, в которых содержится информация о названиях статей, журналов и ФИО авторов
parsePath = 'D:/data/Парсинг данных о разделах/'


# Функция, которая оставляет в списке только уникальные значения
def unique_list(l, exception):
    ulist = []
    if exception != 0:
        [ulist.append(x) for x in l if x not in ulist or x == exception]
    else:
        [ulist.append(x) for x in l if x not in ulist]
    return ulist


# Список русских стоп слов
rus_stop_words = stopwords.words("russian")
# Список ФИО авторов
authors_names_list = ''
# Список названий статей
articles_names_list = []
# Список названий журналов
journal_names_list = []

# Заполнение списков данными из файлов csv
i = -1
while i < 61:
    i += 1
    file = parsePath + GRNTI[i] + '.csv'
    if os.path.exists(file) and os.path.isfile(file):
        with open(file, 'r', newline='', encoding='utf8') as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=['Field1', 'Field2', 'Field3_text'])
            for row in csv_reader:
                if row['Field2'] == 'Field2':
                    journal_names_list.append(str(i))
                    authors_names_list += ' '
                    authors_names_list += str(i)
                    authors_names_list += ' '
                    articles_names_list.append(str(i))
                else:
                    journal_names_list.append(row['Field3_text'])
                    authors_names_list += (row['Field2'])
                    articles_names_list.append(row['Field1'])
    else:
        journal_names_list.append(' ')
        authors_names_list += ' '
        articles_names_list.append(' ')
# Токенизация
authors_names_list = word_tokenize(authors_names_list, "russian")

# Удаление знаков препинаний
tokenizer = RegexpTokenizer(r"\w+")
authors_names_list = tokenizer.tokenize(' '.join(authors_names_list))

# Удаление дублирующихся названий
lst = []
t_ind = 0
i = 0
result = []
for token in authors_names_list:
    if token.isnumeric() and i > t_ind:
        lst = authors_names_list[t_ind:i]
        result += unique_list(lst, 0)
        t_ind = i
    i += 1

lst = []
t_ind = 0
i = 0
result = []
for token in journal_names_list:
    if token.isnumeric() and i > t_ind:
        lst = journal_names_list[t_ind:i]
        result += unique_list(lst, 0)
        t_ind = i
    i += 1

journal_names_list = result

# Обработка полученного текста
# Приведение всех символов к строчному виду

# Длина списков
listLengthJ = len(journal_names_list)
listLengthA = len(articles_names_list)
listLengthAu = len(authors_names_list)

for i in range(0, listLengthJ):
    journal_names_list[i] = journal_names_list[i].lower()

for i in range(0, listLengthA):
    articles_names_list[i] = articles_names_list[i].lower()

for i in range(0, listLengthAu):
    authors_names_list[i] = authors_names_list[i].lower()

t_ind = 0
result = []
for token in authors_names_list:
    if token not in alphabet_rus:
        result.append(token)
    t_ind += 1

authors_names_list = result

# print(authors_names_list)
# print(journal_names_list)
# print(articles_names_list)

# Добавление стоп слов
stop_words_list = 'рис', 'это', 'также', 'которые', 'удк', 'гг', 'однако', 'тыс', 'которых', 'др', 'твой', 'которой', \
    'которого', 'свой', 'твоя', 'этими', 'слишком', 'нами', 'всему', 'будь', 'саму', 'чаще', 'ваше', 'сами', 'наш', \
    'затем', 'еще', 'самих', 'наши', 'ту', 'каждое', 'весь', 'этим', 'наша', 'своих', 'который', 'зато', 'те', 'этих', \
    'вся', 'ваш', 'такая', 'теми', 'ею', 'которая', 'нередко', 'каждая', 'также', 'чему', 'собой', 'самими', 'нем',\
    'вами', 'ими', 'откуда', 'такие', 'тому', 'та', 'очень', 'сама', 'нему', 'алло', 'оно', 'этому', 'кому', 'тобой',\
    'таки', 'твоё', 'каждые', 'твои', 'мой', 'нею', 'самим', 'ваши', 'ваша', 'кем', 'мои', 'однако', 'сразу', 'свое',\
    'ними', 'всё', 'неё', 'тех', 'хотя', 'всем', 'тобою', 'тебе', 'одной', 'другие', 'этого', 'само', 'эта', 'буду',\
    'самой', 'моё', 'своей', 'такое', 'всею', 'будут', 'своего', 'кого', 'свои', 'мог', 'нам', 'особенно', 'её',\
    'самому', 'наше', 'кроме', 'вообще', 'вон', 'мною', 'никто', 'это'

rus_stop_words.extend(stop_words_list)
