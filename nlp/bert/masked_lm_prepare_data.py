class MaskedLmInstance:
  def __init__(self, index, label):
    self.index = index
    self.label = label



def create_masked_lm_predictions(tokens, masked_lm_prob, 
  max_predictions_per_seq, vocab_words, rng):
  """
  Create training data for Masked Language Model tasks.

  @param tokens: input text.
  @param masked_lm_prob: mask probability in masked lm.
  @param max_predictions_per_seq: max number of predictions per sequence.
  @param vocab_words: vocab.
  @param rng: random number generator.
  """
  # stores candidates for masking
  cand_indexes = []
  for (i, token) in enumerate(tokens):
    # skip [CLS] and [SEP] during masking
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append([i])

  rng.shuffle(cand_indexes)
  # stores input sequence after masking, initialized to original seq
  output_tokens = list(tokens)
  num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
  
  # stores masked examples
  masked_lms = []
  # stores processed tokens
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)
    
    masked_token = None
    # 80% probability to replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% probability to not replace with anything
      if rng.random() < 0.5:
        masked_token = tokens[index]
      else:
        # 10% probability to replace with a random token
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token
    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms.sorted(masked_lms, key=lambda x: x.index)

  # stores indexes of mask
  masked_lm_positions = []
  # stores original token
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return output_tokens, masked_lm_positions, masked_lm_labels)
