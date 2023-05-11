import os
import csv
import pickle

from gensim.models import FastText

from fieldnames import GRNTI, fieldnames, fieldnames_symb, alphabet_eng, alphabet_gre, alphabet_latin, key_stopwords, \
    csvfile, symbolsfile
from functionList import pre1, pre2, red
from stopwordsCreation import authors_names_list, journal_names_list, articles_names_list

# Запись заголовков в файл data.csv
with open(csvfile, 'w', encoding='utf8') as csvf:
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()

numAr = 0  # Номер статьи
docs = []  # Список документов
docs_doc2vec = [[]]  # Список списков документов
docs_fasttext = [[""]]  # Список всех документов
rubricks = []  # Список рубрик документов

# Процесс предобработки текста статей до токенизации и записи текста в файл data.csv
for i in range(0, 62):
    directory = 'D:/data/' + GRNTI[i]  # Путь до папки с текстами статей

    symbNum = 0  # Количество символов в рубрике
    n = 0  # Номер статьи в рубрике

    # Создание списка названий журналов для рубрики
    t_indS = 0
    t_indE = 0
    t_ind = 0
    for token in journal_names_list:
        if token == str(i):
            t_indS = t_ind
        if token == str(i + 1):
            t_indE = t_ind
        t_ind += 1
    lstJ = journal_names_list[t_indS + 1: t_indE]

    # Создание списка авторов для рубрики
    au_indS = 0
    au_indE = 0
    au_ind = 0
    for token in authors_names_list:
        if token == str(i):
            au_indS = au_ind
        if token == str(i + 1):
            au_indE = au_ind
        au_ind += 1
    au_lst = authors_names_list[au_indS + 1: au_indE]

    # Создание списка названий для рубрики
    ar_indS = 0
    ar_indE = 0
    ar_ind = 0
    for token in articles_names_list:
        if token == str(i):
            ar_indS = ar_ind
        if token == str(i + 1):
            ar_indE = ar_ind
        ar_ind += 1
    ar_lst = articles_names_list[ar_indS + 1: ar_indE]

    for filename in os.listdir(directory):
        # Путь до файла
        f = os.path.join(directory, filename)
        # Проверка существования файла
        if os.path.isfile(f):
            n += 1
            numAr += 1
            with open(f, 'r', newline='', encoding='utf8') as file:
                text = file.read()

                # Нахождение ключевых слов в тексте
                if 'Ключевые слова' in text:
                    key_start = text.find('Ключевые слова')
                    key_end = text.find('.', key_start)
                    key_start += 15
                    key_words = text[key_start:key_end]
                    key_words = pre1(key_words)
                    for ind in range(0, len(key_stopwords)):
                        key_words = red(key_stopwords[ind], key_words)
                else:
                    key_words = '-'

                # Нахождение аннотации в тексте
                if 'Аннотация' in text:
                    ann_start = text.find('Аннотация')
                    ann_start += 10
                    if 'Abstract' in text:
                        if 0 < text.find('Abstract') - ann_start < len(text) / 2 and text.find(
                                'Ключевые слова') > text.find('Abstract'):
                            ann_end = text.find('Abstract', ann_start)
                        elif 'Ключевые слова' in text:
                            ann_end = text.find('Ключевые слова', ann_start)
                    annotation = text[ann_start:ann_end]
                    # Удаление всех английских слов
                    ew_ind = 0
                    annotation = pre1(annotation)
                    for eng_word in annotation:
                        while ew_ind < len(annotation) and annotation[ew_ind] in alphabet_eng:
                            _index = annotation.find(eng_word)
                            annotation = annotation[:ew_ind] + annotation[ew_ind + len(eng_word):]
                            ew_ind -= 1
                        ew_ind += 1
                    annotation = pre2(annotation)
                else:
                    annotation = '-'
                    if key_words != '-':
                        key_words = pre2(key_words)

                title = ''

                # Начало предобработки текста: шрифт, удаление лишних пробелов, удаление переноса на новую строку, url-ы
                text = pre1(text)

                # Удаление всех английских слов
                ew_ind = 0
                for eng_word in text:
                    while ew_ind < len(text) and text[ew_ind] in alphabet_eng:
                        _index = text.find(eng_word)
                        text = text[:ew_ind] + text[ew_ind + len(eng_word):]
                    ew_ind += 1

                # Удаление всех греческих букв
                gre_ind = 0
                for gre_word in text:
                    while gre_ind < len(text) and text[gre_ind] in alphabet_gre:
                        _index = text.find(gre_word)
                        text = text[:gre_ind] + text[gre_ind + len(gre_word):]
                    gre_ind += 1

                # Удаление всех латинских символов
                lat_ind = 0
                for lat_word in text:
                    while lat_ind < len(text) and text[lat_ind] in alphabet_latin:
                        _index = text.find(lat_word)
                        text = text[:lat_ind] + text[lat_ind + len(lat_word):]
                        lat_ind -= 1
                    lat_ind += 1

                # Удаление из текста названий журналов
                for jn in lstJ:
                    while jn in text and len(jn) > 3:
                        j_index = text.find(jn)
                        text = text[:j_index] + text[j_index + len(jn):]

                # Поиск названий статьи и удаление названий
                for ar in ar_lst:
                    while ar in text and len(ar) > 3:
                        title = ar
                        ar_index = text.find(ar)
                        text = text[:ar_index] + text[ar_index + len(ar):]
                        title = pre2(title)

                text = pre2(text)

                # Удаление из текста ФИО авторов
                au_index = 0
                for word in text:
                    while au_index < len(text) and text[au_index] in au_lst:
                        text.pop(au_index)
                    au_index += 1

                docs.append(text)
                docs_doc2vec.append(text)
                docs_fasttext.append([""])
                for word in text:
                    if len(word) > 1:
                        docs_fasttext[numAr].append(word)
                rubricks.append(directory[8:])

                with open("D:/data/preprocessing_BERT.txt", "a", encoding='utf-8') as fp:
                    for word in text:
                        fp.write(word)
                        fp.write(' ')
                    fp.close()

                print(len(text))

                # print(len(docs_fasttext))

            #     # Запись текста в data.csv
            #     with open(csvfile, 'a', newline='', encoding='utf8') as csvf:
            #         writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            #         symbNum += len(text)
            #         writer.writerow({'Номер': numAr, 'Рубрика': directory[8:], 'Название': title, 'Текст': text,
            #                          'Аннотация': annotation, 'Ключевые_слова': key_words, 'Объем': len(text)})
            #
                # Заполнение файла symbols.csv
            # with open(symbolsfile, 'a', newline='', encoding='utf8') as csvsymb:
            #     writer_symb = csv.DictWriter(csvsymb, fieldnames=fieldnames_symb)
            #     writer_symb.writerow({'Рубрика': directory[8:], 'Количество_статей': n, 'Символы': symbNum})

with open("D:/data/preprocessing.txt", "wb") as fp:
    pickle.dump(docs, fp)

with open("D:/data/preprocessing_doc2vec.txt", "wb") as fp:
    pickle.dump(docs_doc2vec, fp)

with open("D:/data/preprocessing_fasttext.txt", "wb") as fp:
    pickle.dump(docs_fasttext, fp)

with open("D:/data/rubricks.txt", "wb") as fp:
    pickle.dump(rubricks, fp)
