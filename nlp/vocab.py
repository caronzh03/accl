from collections import defaultdict


class Vocab:
  def __init__(self, tokens=None):
    """
      Map input tokens to integers Z(s), where 0 <= Z <= VocabSize.
      Assume tokens is a unique list.
    """
    self.idx_to_token = list()
    self.token_to_idx = dict()

    if tokens is not None:
      if "<unk>" not in tokens:
        tokens = tokens + ["<unk>"]
      for token in tokens:
        self.idx_to_token.append(token)
        self.token_to_idx[token] = len(self.idx_to_token) - 1
      self.unk = self.token_to_idx["<unk>"]
        
  @classmethod
  def build(cls, text, min_freq=1, reserved_tokens=None):
    token_freqs = defaultdict(int)
    for sentence in text:
      for token in sentence:
        token_freqs[token] += 1

    uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
    uniq_tokens += [token for token, freq in token_freqs.items() \
                    if freq > min_freq and token != "<unk>"]
    return cls(uniq_tokens)

  def __len__(self):
    return len(self.idx_to_token)

  def __getitem__(self, token):
    return self.token_to_idx.get(token, self.unk)

  def convert_tokens_to_ids(self, tokens):
    return [self[token] for token in tokens]

  def convert_ids_to_tokens(self, indices):
    return [self.idx_to_token[idx] for idx in indices]



def save_vocab(vocab, path):
  with open(path, 'w') as writer:
    writer.write("\n".join(vocab.idx_to_token))
