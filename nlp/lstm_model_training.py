from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence 
import torch

from vocab import Vocab
from utils import load_sentence_polarity, BowDataset


class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
    super(LSTM, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.linear = nn.Linear(hidden_dim, num_class)

  def forward(self, inputs, lengths):
    """
      lengths: length of each original sequence.
    """
    embeddings = self.embeddings(inputs)
    # lengths must be on CPU if provided as tensor, therefore convert it to list instead
    x_pack = pack_padded_sequence(embeddings, lengths.tolist(), batch_first=True, enforce_sorted=False)
    # hn: final hidden state; cn: final cell state
    hidden, (hn, cn) = self.lstm(x_pack)
    outputs = self.linear(hn[-1])
    probs = F.log_softmax(outputs, dim=1)
    return probs


def collate_fn(examples):
  """
    Process fn for each batch.
    Each example is a tuple from dataset: ([ids], polarity).
  """
  # get length of each sequence  
  lengths = torch.tensor([len(ex[0]) for ex in examples])   
  inputs = [torch.tensor(ex[0]) for ex in examples]
  targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
  # apply padding per batch, such that each batch has the same length
  inputs = pad_sequence(inputs, batch_first=True)
  return inputs, lengths, targets


def main():
  """
    LSTM model training and testing.
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

  # load model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
  model.to(device)

  # train model
  nll_loss = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  model.train()
  for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
      inputs, lengths, targets = [x.to(device) for x in batch]
      log_probs = model(inputs, lengths)
      loss = nll_loss(log_probs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

  # test
  acc = 0
  for batch in tqdm(test_data_loader, desc=f"Testing"):
     inputs, lengths, targets = [x.to(device) for x in batch]
     with torch.no_grad():
       output = model(inputs, lengths)
       acc += (output.argmax(dim=1) == targets).sum().item()

  # Output accuracy
  print(f"Acc: {acc / len(test_data_loader):.2f}")     



if __name__=="__main__":
  main()   
   
