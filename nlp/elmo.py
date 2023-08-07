import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from collections import defaultdict
import numpy
import json
import os

from vocab import Vocab, save_vocab
from utils import load_reuters, get_loader, save_pretrained, \
  BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN


def load_corpus(path, max_tok_len=None, max_seq_len=None):
  """
  Build vocabs from both words and characters.
  @param max_tok_len: max length of each word.
  @param max_seq_len: max length of each sentence.
  """
  # text will store a list of a list of words
  # e.g. [[my, university, is, good], [her, college, is, prestigious]]
  text = []
  charset = {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN}
  with open(path, "r") as f:
    for line in tqdm(f):
      tokens = line.rstrip().split(" ")
      # chop off long input sentence
      if max_seq_len is not None and len(tokens) + 2 > max_seq_len:
        tokens = line[:max_seq_len - 2]
      sent = [BOS_TOKEN]
      for token in tokens:
        # chop off long input word
        if max_tok_len is not None and len(token) + 2 > max_tok_len:
          token = token[:max_tok_len - 2]
        sent.append(token)
        for ch in token:
          charset.add(ch)
      sent.append(EOS_TOKEN)
      text.append(sent)

  # build vocab
  vocab_w = Vocab.build(text, min_freq=2, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
  # build character-level vocab
  vocab_c = Vocab(tokens=list(charset))

  # build word-level corpus
  corpus_w = [vocab_w.convert_tokens_to_ids(sent) for sent in text]
  # build character-level corpus
  corpus_c = []
  bow = vocab_c[BOW_TOKEN]
  eow = vocab_c[EOW_TOKEN]
  for i, sent in enumerate(text):
    sent_c = []
    for token in sent:
      if token == BOS_TOKEN or token == EOS_TOKEN:
        token_c = [bow, vocab_c[token], eow]
      else:
        token_c = [bow] + vocab_c.convert_tokens_to_ids(token) + [eow]
      sent_c.append(token_c)
    corpus_c.append(sent_c)

  return corpus_w, corpus_c, vocab_w, vocab_c


class BiLMDataset(Dataset):
  def __init__(self, corpus_w, corpus_c, vocab_w, vocab_c):
    super(BiLMDataset, self).__init__()
    self.pad_w = vocab_w[PAD_TOKEN]
    self.pad_c = vocab_c[PAD_TOKEN]

    self.data = []
    for sent_w, sent_c in zip(corpus_w, corpus_c):
      self.data.append((sent_w, sent_c))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]

  def collate_fn(self, examples):
    # lengths: batch_size
    seq_lens = torch.LongTensor([len(ex[0]) for ex in examples])

    # inputs_w
    inputs_w = [torch.tensor(ex[0]) for ex in examples]
    inputs_w = pad_sequence(inputs_w, batch_first=True, padding_value=self.pad_w)

    # inputs_c: batch_size * max_seq_len * max_tok_len
    batch_size, max_seq_len = inputs_w.shape
    max_tok_len = max([max([len(tok) for tok in ex[1]]) for ex in examples])

    inputs_c = torch.LongTensor(batch_size, max_seq_len, max_tok_len).fill_(self.pad_c)
    for i, (sent_w, sent_c) in enumerate(examples):
      for j, tok in enumerate(sent_c):
        inputs_c[i][j][:len(tok)] = torch.LongTensor(tok)

    # fw_input_indexes, bw_input_indexes = [], []
    targets_fw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
    targets_bw = torch.LongTensor(inputs_w.shape).fill_(self.pad_w)
    for i, (sent_w, sent_c) in enumerate(examples):
      # ignore <pad> in target
      # <bos>w1w2...wn<eos> => w1w2...wn<eos><pad>
      targets_fw[i][:len(sent_w) - 1] = torch.LongTensor(sent_w[1:])
      # <bos>w1w2...wn<eos> => <pad><bos>w1w2...wn
      targets_bw[i][1:len(sent_w)] = torch.LongTensor(sent_w[:len(sent_w) - 1])

    return inputs_w, inputs_c, seq_lens, targets_fw, targets_bw


class Highway(nn.Module):
  def __init__(self, input_dim, num_layers, activation=F.relu):
    super(Highway, self).__init__()
    self.input_dim = input_dim
    self.layers = torch.nn.ModuleList(
      [nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
    )
    self.activation = activation
    for layer in self.layers:
      layer.bias[input_dim:].data.fill_(1)

  def forward(self, inputs):
    curr_inputs = inputs
    for layer in self.layers:
      projected_inputs = layer(curr_inputs)
      # first half of output is used as current hidden layer's output
      hidden = self.activation(projected_inputs[:, 0:self.input_dim])
      # second half is usd to calculate gate vector
      gate = torch.sigmoid(projected_inputs[:, self.input_dim:])
      # linear interpolation
      curr_intpus = gate * curr_inputs + (1 - gate) * hidden
    return curr_inputs


class ConvTokenEmbedder(nn.Module):
  def __init__(self, vocab_c, char_embedding_dim, char_conv_filters, num_highways, output_dim, pad="<pad>"):
    """
    vocab_c: character-level vocab
    char_embedding_dim: character embedding dimension
    char_conv_filters: convolutional kernels
    num_highways: number of Highway network layers
    """
    super(ConvTokenEmbedder, self).__init__()
    self.vocab_c = vocab_c
    self.char_embeddings = nn.Embedding(len(vocab_c), char_embedding_dim, padding_idx=vocab_c[pad])
    self.char_embeddings.weight.data.uniform_(-0.25, 0.25)

    # construct CNN for each convolutional kernel
    # use conv1d
    self.convolutions = nn.ModuleList()
    for kernel_size, out_channels in char_conv_filters:
      conv = torch.nn.Conv1d(
        in_channels=char_embedding_dim,
        out_channels=out_channels,
        kernel_size=kernel_size,
        bias=True)
      self.convolutions.append(conv)

    # dimension of concatenated filters
    self.num_filters = sum(f[1] for f in char_conv_filters)
    self.num_highways = num_highways
    self.highways = Highway(self.num_filters, self.num_highways, activation=F.relu)

    # since ELMo embedding is a result of linear interpolation of multiple layers,
    # need to ensure each layer's embedding dim to be consistent
    self.projection = nn.Linear(self.num_filters, output_dim, bias=True)

  def forward(self, inputs):
    batch_size, seq_len, token_len = inputs.shape
    inputs = inputs.view(batch_size * seq_len, -1)
    char_embeds = self.char_embeddings(inputs)
    char_embeds = char_embeds.transpose(1, 2)

    conv_hiddens = []
    for i in range(len(self.convolutions)):
      conv_hidden = self.convolutions[i](char_embeds)
      conv_hidden, _ = torch.max(conv_hidden, dim=-1)
      conv_hidden = F.relu(conv_hidden)
      conv_hiddens.append(conv_hidden)

    # concat embeddings obtained from each conv kernel
    token_embeds = torch.cat(conv_hiddens, dim=-1)
    token_embeds = self.highways(token_embeds)
    token_embeds = self.projection(token_embeds)
    token_embeds = token_embeds.view(batch_size, seq_len, -1)
    return token_embeds


class ElmoLstmEncoder(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers):
    super(ElmoLstmEncoder, self).__init__()
    # ensure each intermediate LSTM layer's dim is the same with output & input layers'.
    self.projection_dim = input_dim
    self.num_layers = num_layers

    # forward multi-layer LSTM
    self.forward_layers = nn.ModuleList()
    # forward projection layer: hidden_dim -> self.projection_dim
    self.forward_projections = nn.ModuleList()
    # backward multi-layer LSTM
    self.backward_layers = nn.ModuleList()
    # backward project layer: hidden_dim -> self.projection_dim
    self.backward_projections = nn.ModuleList()

    lstm_input_dim = input_dim
    for _ in range(num_layers):
      # single layer of forward LSTM & projection
      forward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
      forward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)
      # single layer of backward LSTM & projection
      backward_layer = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=1, batch_first=True)
      backward_projection = nn.Linear(hidden_dim, self.projection_dim, bias=True)

      lstm_input_dim = self.projection_dim

      self.forward_layers.append(forward_layer)
      self.forward_projections.append(forward_projection)
      self.backward_layers.append(backward_layer)
      self.backward_projections.append(backward_projection)
    
  def forward(self, inputs, lengths):
    batch_size, seq_len, input_dim = inputs.shape
    # build backward batch's reverse index from forward batch
    rev_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    for i in range(lengths.shape[0]):
      rev_idx[i, :lengths[i]] = torch.arange(lengths[i]-1, -1, -1)
    rev_idx = rev_idx.unsqueeze(2).expand_as(inputs)
    rev_idx = rev_idx.to(inputs.device)
    rev_inputs = inputs.gather(1, rev_idx)

    # forward & backward LSTM inputs
    forward_inputs, backward_inputs = inputs, rev_inputs
    # store each forward & backward hidden layer's state
    stacked_forward_states, stacked_backward_states = [], []

    for layer_index in range(self.num_layers):
      packed_forward_inputs = pack_padded_sequence(forward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
      packed_backward_inputs = pack_padded_sequence(backward_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)

      # calculate forward LSTM
      forward_layer = self.forward_layers[layer_index]
      packed_forward, _ = forward_layer(packed_forward_inputs)
      forward = pad_packed_sequence(packed_forward, batch_first=True)[0]
      forward = self.forward_projections[layer_index](forward)
      stacked_forward_states.append(forward)

      # calculate backward LSTM
      backward_layer = self.backward_layers[layer_index]
      packed_backward, _ = backward_layer(packed_backward_inputs)
      backward = pad_packed_sequence(packed_backward, batch_first=True)[0]
      backward = self.backward_projections[layer_index](backward)
      # restore original sequence's order
      stacked_backward_states.append(backward.gather(1, rev_idx))
    
    return stacked_forward_states, stacked_backward_states



configs = {
    'max_tok_len': 50,
    'train_file': './train.txt', # path to training file, line-by-line and tokenized
    'model_path': './elmo_bilm',
    'char_embedding_dim': 50,
    'char_conv_filters': [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
    'num_highways': 2,
    'projection_dim': 512,
    'hidden_dim': 4096,
    'num_layers': 2,
    'batch_size': 32,
    'dropout_prob': 0.1,
    'learning_rate': 0.0004,
    'clip_grad': 5,
    'num_epoch': 1
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLM(nn.Module):
  def __init__(self, configs, vocab_w, vocab_c):
    super(BiLM, self).__init__()
    self.dropout_prob = configs['dropout_prob']
    # output dim is vocab size
    self.num_classes = len(vocab_w)

    # token embedding
    self.token_embedder = ConvTokenEmbedder(
      vocab_c,
      configs['char_embedding_dim'],
      configs['char_conv_filters'],
      configs['num_highways'],
      configs['projection_dim'])

    # ELMo LSTM encoding
    self.encoder = ElmoLstmEncoder(
      configs['projection_dim'],
      configs['hidden_dim'],
      configs['num_layers'])

    # classifer (output layer)
    self.classifier = nn.Linear(configs['projection_dim'], self.num_classes)

  def forward(self, inputs, lengths):
    token_embeds = self.token_embedder(inputs)
    token_embeds = F.dropout(token_embeds, self.dropout_prob)
    forward, backward = self.encoder(token_embeds, lengths)
    # take forward/backward LSTM's last layer as LM's output
    return self.classifier(forward[-1]), self.classifier(backward[-1])

  def save_pretrained(self, path):
    """
    Save embedding parameters for later ELMo computation.
    """
    os.makedirs(path, exist_ok=True)
    torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pth'))
    torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))



def main():
  # load data
  corpus_w, corpus_c, vocab_w, vocab_c = load_corpus(configs['train_file'])
  train_data = BiLMDataset(corpus_w, corpus_c, vocab_w, vocab_c)
  train_loader = get_loader(train_data, configs['batch_size'])
  
  # cross-entropy loss func
  criterion = nn.CrossEntropyLoss(
    ignore_index=vocab_w[PAD_TOKEN],
    reduction="sum")

  # construct LM
  model = BiLM(configs, vocab_w, vocab_c)
  model.to(device)
  optimizer = optim.Adam(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=configs['learning_rate'])

  # train
  model.train()
  for epoch in range(configs['num_epoch']):
    total_loss = 0
    total_tags = 0 # effective prediction count
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
      inputs_w, inputs_c, seq_lens, targets_fw, targets_bw = [x.to(device) for x in batch]
      optimizer.zero_grad()
      outputs_fw, outputs_bw = model(inputs_c, seq_lens)
      # forward LM loss
      loss_fw = criterion(
        outputs_fw.view(-1, outputs_fw.shape[-1]),
        targets_fw.view(-1))
      # backward LM loss
      loss_bw = criterion(
        outputs_bw.view(-1, outputs_bw.shape[-1]),
        targets_bw.view(-1))
      loss = (loss_fw + loss_bw) / 2.0
      loss.backward()
      # gradient clip
      nn.utils.clip_grad_norm_(model.parameters(), configs['clip_grad'])
      optimizer.step()

      total_loss += loss_fw.item()
      total_tags = seq_lens.sum().item()

    # use forward LM's perplexity as metric
    train_ppl = numpy.exp(total_loss / total_tags)
    print(f"Train PPL: {train_ppl:.2f}")

  # save params
  model.save_pretrained(configs['model_path'])
  json.dump(configs, open(os.path.join(configs['model_path'], 'configs.json'), "w"))
  # save embedding
  save_vocab(vocab_w, os.path.join(configs['model_path'], 'word.dic'))
  save_vocab(vocab_c, os.path.join(configs['model_path'], 'char.dic'))



if __name__ == "__main__":
  main()
