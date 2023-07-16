import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from vocab import Vocab
from utils import load_reuters, get_loader, save_pretrained, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class SGNSDataset(Dataset):
  def __init__(self, corpus, vocab, context_size=2, n_negatives=5, ns_dist=None):
    """
    Skip gram negative sampling dataset.
    """
    self.data = []
    self.bos = vocab[BOS_TOKEN]
    self.eos = vocab[EOS_TOKEN]
    self.pad = vocab[PAD_TOKEN]
    for sentence in tqdm(corpus, desc="Dataset Construction"):
      sentence = [self.bos] + sentence + [self.eos]
      for i in range(1, len(sentence)-1):
        # input: (w, context)
        # output: 0 or 1, indicating if context is a negative sample
        w = sentence[i]
        left_context_index = max(0, i-context_size)
        right_context_index = min(len(sentence), i+context_size)
        context = sentence[left_context_index: i] + sentence[i+1: right_context_index+1]
        context += [self.pad] * (2 * context_size - len(context))
        self.data.append((w, context))

    # number of negative samples
    self.n_negatives = n_negatives
    # negative sampling distro; if ns_dist is None, use average sampling
    ns_dist_present = (ns_dist is not None)
    self.ns_dist = ns_dist if ns_dist_present else torch.ones(len(vocab))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]

  def collate_fn(self, samples):
    words = torch.tensor([sample[0] for sample in samples], dtype=torch.long)
    contexts = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    batch_size, context_size = contexts.shape
    neg_contexts = []
    # negative samples within each batch
    for i in range(batch_size):
      # make sure negative samples does not include current sample's context
      ns_dist = self.ns_dist.index_fill(0, contexts[i], .0)
      neg_contexts.append(torch.multinomial(ns_dist, self.n_negatives * context_size, replacement=True))
    neg_contexts = torch.stack(neg_contexts, dim=0)
    return words, contexts, neg_contexts


class SGNSModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(SGNSModel, self).__init__()
    # current word embedding
    self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
    # context word embedding
    self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)

  def forward_w(self, words):
    w_embeds = self.w_embeddings(words)
    return w_embeds

  def forward_c(self, contexts):
    c_embeds = self.c_embeddings(contexts)
    return c_embeds


def get_unigram_distribution(corpus, vocab_size):
  # calculate Unigram distribution
  # return [freq/total] per token
  token_counts = torch.tensor([0] * vocab_size)
  total_count = 0
  for sentence in corpus:
    total_count += len(sentence)
    for token in sentence:
      token_counts[token] += 1
  unigram_dist = torch.div(token_counts.float(), total_count)
  return unigram_dist


def main():
  # hyperparameters
  embedding_dim = 128
  context_size = 3
  batch_size = 1024
  n_negatives = 5
  num_epoch = 10

  # load data
  corpus, vocab = load_reuters()
  # calculate unigram ditro
  unigram_dist = get_unigram_distribution(corpus, len(vocab))
  # calcualte negative sampling distro baed on unigram distro: p(w)**0.75
  negative_sampling_dist = unigram_dist ** 0.75
  negative_sampling_dist /= negative_sampling_dist.sum()

  dataset = SGNSDataset(corpus, vocab, context_size=context_size, n_negatives=n_negatives, ns_dist=negative_sampling_dist)
  data_loader = get_loader(dataset, batch_size)

  # construct model
  model = SGNSModel(len(vocab), embedding_dim)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # start training
  model.train()
  for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training epoch {epoch}"):
      words, contexts, neg_contexts = [x.to(device) for x in batch]
      optimizer.zero_grad()
      batch_size = words.shape[0]
      # extract word, context, and neg_context from batch
      word_embeds = model.forward_w(words).unsqueeze(dim=2)
      context_embeds = model.forward_c(contexts)
      neg_context_embeds = model.forward_c(neg_contexts)
      # positive sample's classification
      context_loss = F.logsigmoid(torch.bmm(context_embeds, word_embeds).squeeze(dim=2))
      context_loss = context_loss.mean(dim=1)
      # negative sample's classification
      neg_context_loss = F.logsigmoid(torch.bmm(neg_context_embeds, word_embeds).squeeze(dim=2).neg())
      neg_context_loss = neg_context_loss.view(batch_size, -1, n_negatives).sum(dim=2)
      neg_context_loss = neg_context_loss.mean(dim=1)
      # overall loss
      loss = -(context_loss + neg_context_loss).mean()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Total loss: {total_loss:.2f}")

  # combine word embeddings and context embeddings => final embeddings
  combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight
  save_pretrained(vocab, combined_embeds.data, "sgns.vec")

  
if __name__ == "__main__":
  main()
