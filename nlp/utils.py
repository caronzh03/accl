from nltk.corpus import sentence_polarity, treebank
from torch.utils.data import Dataset, DataLoader
import torch

from vocab import Vocab


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
BOW_TOKEN = "<bow>"
EOW_TOKEN = "<eow>"


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


def load_reuters():
  from nltk.corpus import reuters
  # get all sentences in Reuters
  text = reuters.sents()
  # lower-case all words
  text = [[word.lower() for word in sentence] for sentence in text]
  # build word bank
  vocab = Vocab.build(text, reserved_tokens=[BOS_TOKEN, EOS_TOKEN, PAD_TOKEN])
  # convert words to ids
  corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]
  return corpus, vocab



def get_loader(dataset, batch_size):
  """
  Return a DataLoader that loads batches of input samples from dataset.
  """
  return DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=True)



def save_pretrained(vocab, embeds, save_path):
  """
  Save vocab and embeddings obtained from training to a file.
  """
  with open(save_path, "w") as writer:
    # save embeddings' dimension
    writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
    for idx, token in enumerate(vocab.idx_to_token):
      # x is a vector
      vec = " ".join(f"{x}" for x in embeds[idx])
      # each row: (token, its enbeddings)
      # e.g. (token [0.12, 0.45, -9.32, ...])
      writer.write(f"{token} {vec}\n")
