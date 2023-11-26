from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
  def __init__(self, dropout=0.5, embedding_dim=768, num_classes=2):
    super(BertClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(embedding_dim, num_classes)

  def forward(self, input_ids, mask):
    """
    @param input_ids: in shape of (batch_size, max_sequence_length).
    @param mask: in shape of (batch_size, max_sequence_length).
    @return logits in shape of (batch_size, num_classes), in this case num_classes=5.
    """
    # ignore first output, which is the embedding vectors of all tokens
    # in a sequence; only care about second output, which is the embedding
    # vector of [CLS] token, which will be used as input to classifier
    _, pooled_output = self.bert(input_ids=input_ids, attention_mask=mask,
      return_dict=False)
    dropout_output = self.dropout(pooled_output)
    linear_output = self.linear(dropout_output)
    return linear_output

