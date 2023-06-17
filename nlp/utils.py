from nltk.corpus import sentence_polarity
from torch.utils.data import Dataset

from vocab import Vocab


class BowDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]


def load_sentence_polarity():
  vocab = Vocab.build(sentence_polarity.sents())
  # training data
  pos_train_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')[:4000]]
  neg_train_data = [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[:4000]]
  train_data = pos_train_data + neg_train_data

  # test data
  pos_test_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')[4000:]]
  neg_test_data = [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[4000:]]
  test_data = pos_test_data + neg_test_data

  return train_data, test_data, vocab
