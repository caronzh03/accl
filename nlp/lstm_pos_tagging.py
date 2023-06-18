from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence 
import torch

from vocab import Vocab
from utils import load_treebank, BowDataset


class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
    super(LSTM, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.output = nn.Linear(hidden_dim, num_class)

  def forward(self, inputs, lengths):
    embeddings = self.embeddings(inputs)
    x_pack = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
    hidden, (hn, cn) = self.lstm(x_pack)
    # opposite to pack_padded_sequence, pad_packed_sequence unpacks x_pack
    hidden, _ = pad_packed_sequence(hidden, batch_first=True)
    # need to use all hidden layer's states
    outputs = self.output(hidden)
    log_probs = F.log_softmax(outputs, dim=-1)
    return log_probs


def main():
  """
    LSTM Part-of-Speech (PoS) training and testing.
  """  
  from tqdm.auto import tqdm
  
  # hyperparameter
  embedding_dim = 128
  hidden_dim = 256
  batch_size = 32
  num_epoch = 5

  # load data
  train_data, test_data, vocab, tag_vocab = load_treebank()
  num_class = len(tag_vocab)

  def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    # each tag corresponds to one result
    targets = [torch.tensor(ex[1]) for ex in examples]
    # pad both inputs and outputs
    inputs = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    targets = pad_sequence(targets, batch_first=True, padding_value=vocab["<pad>"])

    # mask records effective tags
    mask = (inputs != vocab["<pad>"])
    return inputs, lengths, targets, mask 

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
      inputs, lengths, targets, mask = [x.to(device) for x in batch]
      log_probs = model(inputs, lengths)
      loss = nll_loss(log_probs[mask], targets[mask])
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

  # test
  acc = 0
  total = 0
  for batch in tqdm(test_data_loader, desc=f"Testing"):
     inputs, lengths, targets, mask = [x.to(device) for x in batch]
     with torch.no_grad():
       output = model(inputs, lengths)
       acc += (output.argmax(dim=-1) == targets)[mask].sum().item()
       total += mask.sum().item()

  # Output accuracy
  print(f"Acc: {acc / total:.2f}")   


if __name__=="__main__":
  main()   
