import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from vocab import Vocab
from utils import load_reuters, get_loader, save_pretrained, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NGramDataset(Dataset):
  def __init__(self, corpus, vocab, context_size=2):
    """
    Provides dataset accessors for Feed-forward NN LM.
    """
    self.data = []
    self.bos = vocab[BOS_TOKEN]
    self.eos = vocab[EOS_TOKEN]
    for sentence in tqdm(corpus, desc="Dataset construction"):
      # insert bos, eos tokens
      sentence = [self.bos] + sentence + [self.eos]
      # skip sentences that have shorter length than context_size
      if (len(sentence) < context_size):
        continue
      for i in range (context_size, len(sentence)):
        # model input: context_size-lengthed sequence before current token
        context = sentence[i - context_size : i]
        # model output: current token
        target = sentence[i]
        # training example: (context, target)
        self.data.append((context, target))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]

  def collate_fn(self, examples):
    # construct batched input/output from individual samples in self.data
    # and convert them to tensors
    inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long) 
    return (inputs, targets)


    
class FeedForwardNNLM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
    super(FeedForwardNNLM, self).__init__()
    # word embedding
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    # linear transformation: embedding -> hidden layer
    self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
    # ReLU activation
    self.activate = F.relu
    # linear transformation: hidden layer -> output
    self.linear2 = nn.Linear(hidden_dim, vocab_size)
    
  def forward(self, inputs):
    # project input sequence into embeddings, use view function to reshape the tensors
    # to flatten them
    embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
    hidden = self.activate(self.linear1(embeds))
    output = self.linear2(hidden)
    # calculate distributed probability from output logits
    log_probs = F.log_softmax(output, dim=1)
    return log_probs



def main():
  # hyperparameters
  embedding_dim = 128
  hidden_dim = 256
  batch_size = 1024
  context_size = 3
  num_epoch = 10

  # read training data
  corpus, vocab = load_reuters()
  dataset = NGramDataset(corpus, vocab, context_size)
  data_loader = get_loader(dataset, batch_size)

  # use negative log likelihood loss
  nll_loss = nn.NLLLoss()

  # construct model
  model = FeedForwardNNLM(len(vocab), embedding_dim, context_size, hidden_dim)
  model.to(device)

  # use Adam optimizer
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  model.train()
  total_losses = []
  for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
      inputs, targets = [x.to(device) for x in batch]
      optimizer.zero_grad()
      log_probs = model(inputs)
      loss = nll_loss(log_probs, targets)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    print(f"Loss: {total_loss:.2f}")
    total_losses.append(total_loss)

  # save model.embeddings to ffnnlm.vec file
  save_pretrained(vocab, model.embeddings.weight.data, "ffnnlm.vec")

    

if __name__=="__main__":
  main()   
