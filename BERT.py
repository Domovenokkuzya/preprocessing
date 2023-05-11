from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizerFast
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import BertConfig, BertForMaskedLM
from transformers import Trainer

bert_wordpiece_tokenizer = BertWordPieceTokenizer()

bert_wordpiece_tokenizer.train("D:\data\preprocessing_BERT.txt")

bert_wordpiece_tokenizer.get_vocab()

bert_wordpiece_tokenizer.save_model("D:/data/")

# tokenized_sentence = tokenizer.encode("цифровизация")
# Пример
# print(tokenized_sentence.tokens)

tokenizer = BertTokenizerFast.from_pretrained("D:/data/")

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path="D:\data\preprocessing_BERT.txt", block_size=128)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="BERT",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=128)

bert = BertForMaskedLM(BertConfig())

trainer = Trainer(model=bert,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=dataset)

trainer.train()

trainer.save_model("D:\data\BERT1")


