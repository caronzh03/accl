import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from vocab import Vocab
from utils import load_reuters, get_loader, save_pretrained, BOS_TOKEN, EOS_TOKEN


class CbowDataset(Dataset):
  def __init__(self, corpus, vocab, context_size=2):
    self.data = []
    self.bos = vocab[BOS_TOKEN]
    self.eos = vocab[EOS_TOKEN]
    for sentence in tqdm(corpus, desc="Dataset Construction"):
      # sentence is a vector, whose element corresponds to the index of each token in vocab
      # [1, 19829, 1445, 136, 137...]
      sentence = [self.bos] + sentence + [self.eos]
      # if sentence length is not long enough to build a training sample, skip
      if len(sentence) < context_size * 2 + 1:
        continue
      for i in range(context_size, len(sentence) - context_size):
        # input: take context_size-lengthed tokens before & after current token
        context = sentence[i-context_size: i] + sentence[i+1: i+context_size+1]
        # output: current token
        target = sentence[i]
        self.data.append((context, target))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]

  def collate_fn(self, samples):
    inputs = torch.tensor([pair[0] for pair in samples])
    targets = torch.tensor([pair[1] for pair in samples])
    return (inputs, targets)


class CbowModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(CbowModel, self).__init__()
    # embeddings
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    # output
    self.output = nn.Linear(embedding_dim, vocab_size, bias=False)

  def forward(self, inputs):
    # inputs is a list of context defined by Dataset class
    embeds = self.embeddings(inputs)
    # hidden layer: average
    hidden = embeds.mean(dim=1)
    output = self.output(hidden)
    log_probs = F.log_softmax(output, dim=1)
    return log_probs


def main():
  # hyper parameters
  batch_size = 512
  num_epoch = 10
  embedding_dim = 128

  # load data
  corpus, vocab = load_reuters()
  dataset = CbowDataset(corpus, vocab)
  dataloader = get_loader(dataset, batch_size)

  # define loss function
  nll_loss = nn.NLLLoss()

  # construct model
  model = CbowModel(len(vocab), embedding_dim)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  # define optimizer
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # start training
  model.train()
  for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training in epoch {epoch}"):
      inputs, targets = [x.to(device) for x in batch]
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = nll_loss(outputs, targets)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Total loss: {total_loss:.2f}")  

  # save pretrained embeddings
  save_pretrained(vocab, model.embeddings.weight.data, "cbow.vec")



if __name__ == "__main__":
  main()
