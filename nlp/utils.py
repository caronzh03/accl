from nltk.corpus import sentence_polarity, treebank
from torch.utils.data import Dataset
import torch

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


def load_treebank():
  # sents stores tagged sentences
  # postags stores tagged result
  sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

  # "<pad>" is a placeholder to make all inputs equal-length
  vocab = Vocab.build(sents, reserved_tokens=["<pad>"])
  tag_vocab = Vocab.build(postags)

  # training set
  train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) \
               for sentence, tags in zip(sents[:3000], postags[:3000])]
  # test set
  test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) \
               for sentence, tags in zip(sents[3000:], postags[3000:])]
  
  return train_data, test_data, vocab, tag_vocab


def length_to_mask(lengths, device):
  """
    Convert input sequence's length to Mask matrix.

    >>> lengths = torch.tensor([3, 5, 4])
    >>> length_to_mask(lengths)
    >>> tensor([[True, True, True, False, False],
                [True, True, True, True, True],
                [True, True, True, True, False]])
    :param lengths: [batch,]
    :return: batch * max_len
  """
  max_len = torch.max(lengths)
  mask = torch.arange(max_len).to(device).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
  return mask

