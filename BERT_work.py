import textwrap

import torch
from torch import cosine_similarity
from transformers import BertTokenizerFast, BertModel, LayoutLMv2FeatureExtractor, LayoutLMv2Processor

bert = BertModel.from_pretrained(r"D:/data/BERT1/")
tokenizer = BertTokenizerFast.from_pretrained("D:/data/")

with open("C:/Users/user/PycharmProjects/diplom_new/diplom/database/txt/Automatisation.txt", "r",
          encoding='utf8') as fp:  # Unpickling
    s1 = fp.read()

with open("C:/Users/user/PycharmProjects/diplom_new/diplom/database/txt/История1.txt", "r",
          encoding='utf8') as fp:  # Unpickling
    s2 = fp.read()

text1 = textwrap.wrap(s1, 512)

text2 = textwrap.wrap(s2, 512)

print(len(text1), len(text2))


def doc(tokenizer, model, s):
    sum = 0
    tens = 0
    s_l = s[0]
    s_l = tokenizer.encode(s_l)
    s_l = torch.tensor(s_l)
    # print("2",s1) # prints tensor([ 101, 7592, ...
    s_l = s_l.unsqueeze(
        0)  # add an extra dimension, why ? the model needs to be fed in batches, we give a dummy batch 1
    # print("3",s1) # prints tensor([[ 101, 7592,
    with torch.no_grad():
        output_1 = model(s_l)
    logits_s = output_1[0]
    logits_s = torch.squeeze(logits_s)
    s_l = logits_s.reshape(1, logits_s.numel())
    tens = s_l

    for chunk in s:
        chunk = tokenizer.encode(chunk)
        chunk = torch.tensor(chunk)
        # print("2",s1) # prints tensor([ 101, 7592, ...
        chunk = chunk.unsqueeze(
            0)  # add an extra dimension, why ? the model needs to be fed in batches, we give a dummy batch 1
        # print("3",s1) # prints tensor([[ 101, 7592,
        with torch.no_grad():
            output_1 = model(chunk)
        logits_s = output_1[0]
        logits_s = torch.squeeze(logits_s)
        a = logits_s.reshape(1, logits_s.numel())
        tens = torch.cat((tens, a), 1)
    return tens


a = doc(tokenizer, bert, text1)
print(1)
b = doc(tokenizer, bert, text2)

print(a.shape, b.shape)

if a.shape[1] < b.shape[1]:
    pad_size = (0, b.shape[1] - a.shape[1])
    a = torch.nn.functional.pad(a, pad_size, mode='constant', value=0)
else:
    pad_size = (0, a.shape[1] - b.shape[1])
    b = torch.nn.functional.pad(b, pad_size, mode='constant', value=0)

cos_sim = cosine_similarity(a, b)

print("got cosine similarity", cos_sim)
