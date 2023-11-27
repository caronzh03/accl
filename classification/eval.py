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

        acc = (output.argmax(dim=1) == test_label).sum().item()
        total_acc_test += acc

    print(f"Test accuracy: {total_acc_test / len(eval_data): .3f}")

