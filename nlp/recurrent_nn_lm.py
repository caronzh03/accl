import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from vocab import Vocab
from utils import load_reuters, get_loader, save_pretrained, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RnnlmDataset(Dataset):
  def __init__(self, corpus, vocab):
    self.data = []
    self.bos = vocab[BOS_TOKEN]
    self.eos = vocab[EOS_TOKEN]
    self.pad = vocab[PAD_TOKEN]
    for sentence in tqdm(corpus, desc="Dataset Construction"):
      # input: <bos>, w1, w2, ... wn
      input = [self.bos] + sentence
      # output: w1, w2, ..., wn, <eos>
      output = sentence + [self.eos]
      self.data.append((input, output))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]

  def collate_fn(self, samples):
    # samples:[(input1, output1) ... (inputN, outputN)]
    inputs = [torch.tensor(sample[0]) for sample in samples]
    outputs = [torch.tensor(sample[1]) for sample in samples]
    # pad all sequences from inputs to the same length
    inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad)
    # pad all sequences from outputs to the same length
    outputs = pad_sequence(outputs, batch_first=True, padding_value=self.pad)
    return (inputs, outputs)

  

class RNNLM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    super(RNNLM, self).__init__()
    # word embedding
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    # rnn, use LSTM here
    self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    # output
    self.output = nn.Linear(hidden_dim, vocab_size)

  def forward(self, inputs):
    embeds = self.embeddings(inputs)
    # hidden layer at each time t
    hidden, _ = self.rnn(embeds)
    output = self.output(hidden)
    log_probs = F.log_softmax(output, dim=2)
    return log_probs

  

def main():
  # hyperparameters
  # since input/output length is large,
  # should adjust accordingly to fit in GPU memory
  batch_size = 64
  embedding_dim = 128
  hidden_dim = 256
  num_epoch = 10

  # load data
  corpus, vocab = load_reuters()
  dataset = RnnlmDataset(corpus, vocab)
  data_loader = get_loader(dataset, batch_size)

  # define loss function
  # ignore PAD_TOKEN's loss
  nll_loss = nn.NLLLoss(ignore_index=dataset.pad)

  # construct model
  model = RNNLM(len(vocab), embedding_dim, hidden_dim)
  model.to(device)

  # define optimizer
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # start training
  model.train()
  for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
      inputs, outputs = [x.to(device) for x in batch]
      optimizer.zero_grad()
      log_probs = model(inputs)
      loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]), outputs.view(-1))
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Loss: {total_loss: .2f}")

  # save embeddings
  save_pretrained(vocab, model.embeddings.weight.data, "rnnlm.vec")



if __name__ == "__main__":
  main()
