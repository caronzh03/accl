from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import torch

from bert_model import BertClassifier
from dataset import SSTTestDataset


class Tester(object):
  def __init__(self, model, batch_size):
    self.model = model
    self.batch_size = batch_size
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

  def test(self, test_data):
    test = SSTTestDataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=self.batch_size)
    self.model = self.model.to(self.device)

    self.model.eval()
    i = 0
    with torch.no_grad():
      for test_input, original in tqdm(test_dataloader, desc="Test"):
        input_id = test_input['input_ids'].squeeze(1).to(self.device)
        mask = test_input['attention_mask'].squeeze(1).to(self.device)

        output = self.model(input_id, mask)
        predicted_labels = output.argmax(dim=1)
        i += 1
        if (i + 1) % 10 == 0:
          for sents, labels in zip(original, predicted_labels):
            print(f"{sents}: {labels}")


def main():
  # sample test dataset
  test_datapath = "/media/tianlu/SSD/datasets/stanford-sentiment-treebank/test.csv"
  df_test = pd.read_csv(test_datapath)
  print(f"test datasize: {len(df_test)}")

  # BERT model
  bert_classifier = BertClassifier()
  model_path = "models/best_bert_classifier.pt"
  bert_classifier.load_state_dict(torch.load(model_path))

  tester = Tester(bert_classifier, 16)
  tester.test(df_test)


def spotcheck():
  # note: train/test data has quote and `b` character in original sentence, removing it
  # during inference will incorrectly tokenize the text input
  sentence = "b\"may be far from the best of the series , but it 's assured , wonderfully respectful of its past and thrilling enough to make it abundantly clear that this movie phenomenon has once again reinvented itself for a new generation . \""
  print(sentence)
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  inputs = tokenizer(sentence, padding='max_length', max_length=512,
                     truncation=True, return_tensors="pt")


  bert_classifier = BertClassifier()
  model_path = "models/best_bert_classifier.pt"
  bert_classifier.load_state_dict(torch.load(model_path))
  bert_classifier = bert_classifier.to("cuda")
  bert_classifier.eval()

  output = bert_classifier(inputs['input_ids'].to("cuda"), inputs['attention_mask'].to("cuda"))
  print(output.argmax(dim=1))


if __name__ == "__main__":
  main()

