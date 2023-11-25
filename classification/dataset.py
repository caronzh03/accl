import torch
import numpy as np
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class SSTDataset(torch.utils.data.Dataset):
  def __init__(self, df):
    """
    df: dataset with columns 'target' and 'sentence',
        where 'target' is of value 0 or 1, denoting positive/negative
        sentiment of a sentence.
    """
    self.labels = list(df['target'])
    self.texts = [tokenizer(text, padding='max_length', max_length=512,
                            truncation=True, return_tensors="pt")
                            for text in df['sentence']]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.texts[idx], np.array(self.labels[idx])


class BBCDataset(torch.utils.data.Dataset):
  def __init__(self, df):
    """
    df: dataset with columns 'category' and 'text', where 'category'
    is one of: business, entertainment, sport, tech, or politics,
    denoting the topic category of a bbc news excerpt.
    """
    labels = {
      'business': 0,
      'entertainment': 1,
      'sport': 2,
      'tech': 3,
      'politics': 4
    }
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

