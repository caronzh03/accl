import torch
from transformers import BertTokenizer
import pandas as pd
import numpy as np

from bert_model import BertClassifier


# sample test dataset
test_datapath = "/media/tianlu/SSD/datasets/stanford-sentiment-treebank/test.csv"
df = pd.read_csv(test_datapath)
df_test = df.sample(frac=0.1)
sents = df_test['sentence']

# tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# BERT model
bert_classifier = BertClassifier()
model_path = "models/best_bert_classifier.pt"
bert_classifier.load_state_dict(torch.load(model_path))
bert_classifier = bert_classifier.to("cuda")
bert_classifier.eval()

# inference
for sent in sents:
  bert_input = tokenizer(sent, padding='max_length', max_length=512,
    truncation=True, return_tensors="pt")
  input_ids = bert_input["input_ids"].squeeze(1).to("cuda")
  mask = bert_input["attention_mask"].squeeze(1).to("cuda")
  output = bert_classifier(input_ids=input_ids, mask=mask)
  predicted_label = output.argmax(dim=1).item()
  print(f"{predicted_label}: {sent}")

