import torch
import numpy as np
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = {
  'business': 0,
  'entertainment': 1,
  'sport': 2,
  'tech': 3,
  'politics': 4
}


class Dataset(torch.utils.data.Dataset):
  def __init__(self, df):
    # convert text label to idx label
    self.labels = [labels[label] for label in df['category']]
    # convert texts to tokenized inputs:
    # {input_ids: [], attention_mask: [], token_type_ids: []}
    self.texts = [tokenizer(text, padding='max_length', max_length=512,
                            truncation=True, return_tensors="pt")
                            for text in df['text']]

  def __len__(self):
    return len(self.labels)

  def get_batch_labels(self, idx):
    return np.array(self.labels[idx])

  def get_batch_texts(self, idx):
    return self.texts[idx]

  def __getitem__(self, idx):
    batch_texts = self.get_batch_texts(idx)
    batch_y = self.get_batch_labels(idx)
    return batch_texts, batch_y


                    
