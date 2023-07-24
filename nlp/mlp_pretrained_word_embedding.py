from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch

from vocab import Vocab
from utils import load_sentence_polarity, BowDataset


class MLP(nn.Module):
  def __init__(self, vocab, pt_vocab, pt_embeddings, hidden_dim, num_class):
    """
    Use pretrained word embeddings instead of randomly initialized Embeddings.
    """
    super(MLP, self).__init__()
    # use pretrained vocab & embeddings to initialize embedding layer
    vocab_size = len(vocab)
    embedding_dim = pt_embeddings.shape[1]
    self.embeddings = nn.EmbeddingBag(vocab_size, embedding_dim)
    self.embeddings.weight.data.uniform_(-0.1, 0.1)
    for idx, token in enumerate(vocab.idx_to_token):
      pt_idx = pt_vocab[token]
      # only initialize tokens seen during pretraining
      # for unknown tokens, keep randomly initialized value
      if pt_idx != pt_vocab.unk:
        self.embeddings.weight[idx].data.copy_(pt_embeddings[pt_idx])

    # linear transformation: word embedding -> hidden layer
    self.linear1 = nn.Linear(embedding_dim, hidden_dim)
    # ReLU activation
    self.activate = F.relu
    # linear transformation: activation layer -> output layer
    self.linear2 = nn.Linear(hidden_dim, num_class)

  def forward(self, inputs, offsets):
    embeddings = self.embeddings(inputs, offsets)
    hidden = self.activate(self.linear1(embeddings))
    outputs = self.linear2(hidden)
    probs = F.log_softmax(outputs, dim=1)
    return probs
    

def collate_fn(examples):
  """
    Process fn for each batch.
    Each example is a tuple: ([ids], polarity).
  """
  inputs = [torch.tensor(ex[0]) for ex in examples]
  targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
  offsets = [0] + [i.shape[0] for i in inputs]
  offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
  inputs = torch.cat(inputs)
  return inputs, offsets, targets


def load_pretrained(load_path):
  """
  Load pretrained word embeddings in format:
  <token> [embeddings]
  """
  with open(load_path, "r") as fin:
    n, d = map(int, fin.readline().split())
    tokens = []
    embeds = []
    for line in fin:
      line = line.rstrip().split(' ')
      token, embed = line[0], list(map(float, line[1:]))
      tokens.append(token)
      embeds.append(embed)

    pt_vocab = Vocab(tokens)
    pt_embeds = torch.tensor(embeds, dtype=torch.float)

  return pt_vocab, pt_embeds


def main():
  """
    MLP model training and testing.
  """  
  from tqdm.auto import tqdm
  
  # hyperparameter
  embedding_dim = 128
  hidden_dim = 256
  num_class = 2
  batch_size = 32
  num_epoch = 5

  # load data
  train_data, test_data, vocab = load_sentence_polarity()
  train_dataset = BowDataset(train_data)
  test_dataset = BowDataset(test_data)
  train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
  test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

  # load model & initialize with pretrained word embeddings
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  pt_vocab, pt_embeds = load_pretrained("glove.vec")
  model = MLP(vocab, pt_vocab, pt_embeds, hidden_dim, num_class)
  model.to(device)

  # train model
  nll_loss = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  model.train()
  for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
      inputs, offsets, targets = [x.to(device) for x in batch]
      log_probs = model(inputs, offsets)
      loss = nll_loss(log_probs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

  # test
  acc = 0
  for batch in tqdm(test_data_loader, desc=f"Testing"):
     inputs, offsets, targets = [x.to(device) for x in batch]
     with torch.no_grad():
       output = model(inputs, offsets)
       acc += (output.argmax(dim=1) == targets).sum().item()

  # Output accuracy
  print(f"Acc: {acc / len(test_data_loader):.2f}")     



if __name__=="__main__":
  main()   
   
