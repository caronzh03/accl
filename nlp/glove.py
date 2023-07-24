import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from collections import defaultdict

from vocab import Vocab
from utils import load_reuters, get_loader, save_pretrained, BOS_TOKEN, EOS_TOKEN


class GloveDataset(Dataset):
  def __init__(self, corpus, vocab, context_size=2):
    # record co-occurrence of current word with context
    self.cooccur_counts = defaultdict(float)
    self.bos = vocab[BOS_TOKEN]
    self.eos = vocab[EOS_TOKEN]
    for sentence in tqdm(corpus, desc="Dataset Construction"):
      sentence = [self.bos] + sentence + [self.eos]
      for i in range(1, len(sentence)-1):
        w = sentence[i]
        left_contexts = sentence[max(0, i - context_size): i]
        right_contexts = sentence[i + 1: min(len(sentence), i + context_size) + 1]
        # co-occurence declines as distance increases: 1/d(w,c)
        for k, c in enumerate(left_contexts[::-1]):
          self.cooccur_counts[(w, c)] += 1 / (k + 1)
        for k, c in enumerate(right_contexts):
          self.cooccur_counts[(w, c)] += 1 / (k + 1)
    self.data = [(w, c, count) for (w, c), count in self.cooccur_counts.items()]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]

  def collate_fn(self, examples):
    words = torch.tensor([ex[0] for ex in examples])
    contexts = torch.tensor([ex[1] for ex in examples])
    counts = torch.tensor([ex[2] for ex in examples])
    return (words, contexts, counts)


class GloveModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(GloveModel, self).__init__()
    # word emebddings & biases
    self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.w_biases = nn.Embedding(vocab_size, 1)
    # contexts & biases
    self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.c_biases = nn.Embedding(vocab_size, 1)

  def forward_w(self, words):
    w_embeds = self.w_embeddings(words)
    w_biases = self.w_biases(words)
    return w_embeds, w_biases

  def forward_c(self, contexts):
    c_embeds = self.c_embeddings(contexts)
    c_biases = self.c_biases(contexts)
    return c_embeds, c_biases


def main():
  # hyperparameters: sample weights calculation
  m_max = 100
  alpha = 0.75
  context_size = 3
  batch_size = 512
  embedding_dim = 128
  num_epoch = 10

  # load data
  corpus, vocab = load_reuters()
  dataset = GloveDataset(corpus, vocab, context_size=context_size)
  data_loader = get_loader(dataset, batch_size)

  # build model
  model = GloveModel(len(vocab), embedding_dim)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # train
  model.train()
  for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training in epoch {epoch}"):
      words, contexts, counts = [x.to(device) for x in batch]
      word_embeds, word_biases = model.forward_w(words)
      context_embeds, context_biases = model.forward_c(words)
      log_counts = torch.log(counts)
      # sample weight
      weight_factor = torch.clamp(torch.pow(counts / m_max, alpha), max=1.0)
      optimizer.zero_grad()
      # L2 loss
      loss = (torch.sum(word_embeds * context_embeds, dim=1) + word_biases + context_biases - log_counts) ** 2
      # weighted loss
      weighted_loss = (weight_factor * loss).mean()
      weighted_loss.backward()
      optimizer.step()
      total_loss += weighted_loss.item()
    print(f"Loss: {total_loss:.2f}")

  # combine word embeddings and context embeddings => final embeddings
  combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight
  save_pretrained(vocab, combined_embeds.data, "glove.vec")



if __name__ == "__main__":
  main()
