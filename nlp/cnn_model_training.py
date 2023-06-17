from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from vocab import Vocab
from utils import load_sentence_polarity, BowDataset


class CNN(nn.Module):
  def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
    """
      filter_size: kernel dimension.
      num_filter: number of kernels.
    """
    super(CNN, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    # padding=1: pad 1 input before an after each sequence  
    self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
    self.activate = F.relu
    self.linear = nn.Linear(num_filter, num_class)

  def forward(self, inputs):
    embedding = self.embedding(inputs)
    convolution = self.activate(self.conv1d(embedding.permute(0, 2, 1)))
    pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
    outputs = self.linear(pooling.squeeze(dim=2))
    probs = F.log_softmax(outputs, dim=1)
    return probs


def collate_fn(examples):
  """
    Process fn for each batch.
    Each example is a tuple from dataset: ([ids], polarity).
  """
  inputs = [torch.tensor(ex[0]) for ex in examples]
  targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
  # apply padding per batch, such that each batch has the same length
  inputs = pad_sequence(inputs, batch_first=True)
  return inputs, targets


def main():
  """
    CNN model training and testing.
  """  
  from tqdm.auto import tqdm
  
  # hyperparameter
  embedding_dim = 128
  filter_size = 3
  num_filter = 100
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
  model = CNN(len(vocab), embedding_dim, filter_size, num_filter, num_class)
  model.to(device)

  # train model
  nll_loss = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  model.train()
  for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
      inputs, targets = [x.to(device) for x in batch]
      log_probs = model(inputs)
      loss = nll_loss(log_probs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")

  # test
  acc = 0
  for batch in tqdm(test_data_loader, desc=f"Testing"):
     inputs, targets = [x.to(device) for x in batch]
     with torch.no_grad():
       output = model(inputs)
       acc += (output.argmax(dim=1) == targets).sum().item()

  # Output accuracy
  print(f"Acc: {acc / len(test_data_loader):.2f}")     



if __name__=="__main__":
  main()   
   
