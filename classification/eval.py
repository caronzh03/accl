import torch
from tqdm import tqdm

from dataset import SSTDataset


class Evaler(object):
  def __init__(self, model, batch_size):
    self.model = model
    self.batch_size = batch_size
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

  def eval(self, eval_data):
    test = SSTDataset(eval_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=self.batch_size)
    self.model = self.model.to(self.device)

    total_acc_test = 0
    self.model.eval()
    with torch.no_grad():
      for test_input, test_label in tqdm(test_dataloader, desc="Eval"):
        test_label = test_label.to(self.device)
        mask = test_input['attention_mask'].squeeze(1).to(self.device)
        input_id = test_input['input_ids'].squeeze(1).to(self.device)

        output = self.model(input_id, mask)
        predicted_label = output.argmax(dim=1)

        acc = (predicted_label == test_label).sum().item()
        total_acc_test += acc

    print(f"Test accuracy: {total_acc_test / len(eval_data): .3f}")


def main():
  import pandas as pd
  import numpy as np

  from bert_model import BertClassifier

  test_data = "/media/tianlu/SSD/datasets/stanford-sentiment-treebank/train.csv"
  df = pd.read_csv(test_data)
  df_eval, _ = np.split(df.sample(frac=1, random_state=42),
                        [int(0.05 * len(df))])
  print(f"eval datasize: {len(df_eval)}")

  model = BertClassifier()
  model_path = "models/best_bert_classifier.pt"
  model.load_state_dict(torch.load(model_path))

  evaler = Evaler(model, 16)
  evaler.eval(df_eval)


if __name__ == "__main__":
  main()

