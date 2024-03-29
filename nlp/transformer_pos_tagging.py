import math
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence 
import torch

from vocab import Vocab
from utils import load_treebank, BowDataset, length_to_mask


class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=0.1, max_len=512):
    super(PositionalEncoding, self).__init__()

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    # encode even positions
    pe[:, 0::2] = torch.sin(position * div_term)
    # encode odd positions 
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    # do not compute gradient for positional encoding layer
    self.register_buffer('pe', pe)

  def forward(self, x):
    # add word embedding with positional embedding
    x = x + self.pe[:x.size(0), :]
    return x


class Transformer(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class,
               dim_feedforward=512, num_head=2, num_layers=2, dropout=0.1,
               max_len=512, activation: str = "relu"):
    super(Transformer, self).__init__()
    self.embedding_dim = embedding_dim
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    # positional encoding
    self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
    # encoder
    encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    # output
    self.output = nn.Linear(hidden_dim, num_class)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def forward(self, inputs, lengths):
    """
      lengths: length of each original sequence.
    """
    # input's 1st dimension is batch_size, need to transpose input shape to what TransformerEncoder
    # needs: (length, batch_size)    
    inputs = torch.transpose(inputs, 0, 1)  
    hidden_states = self.embeddings(inputs)
    hidden_states = self.position_embedding(hidden_states)
    attention_mask = length_to_mask(lengths, self.device) == False
    # restore to batch_first
    hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask).transpose(0, 1)
    # take each input's hidden layer
    logits = self.output(hidden_states)
    probs = F.log_softmax(logits, dim=-1)
    return probs


def main():
  """
    Transformer model training and testing.
  """  
  from tqdm.auto import tqdm
  
  # hyperparameter
  embedding_dim = 128
  hidden_dim = 128
  batch_size = 32
  num_epoch = 5

  # load data
  train_data, test_data, vocab, tag_vocab = load_treebank()
  num_class = len(tag_vocab)
  train_dataset = BowDataset(train_data)
  test_dataset = BowDataset(test_data)

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

  train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
  test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

  # load model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Transformer(len(vocab), embedding_dim, hidden_dim, num_class)
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
   
