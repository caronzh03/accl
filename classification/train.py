import argparse
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from bert_model import BertClassifier
from dataset import SSTDataset


class Trainer(object):
  def __init__(self, model, batch_size, learning_rate, epochs):
    # hyperparameters
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.epochs = epochs
    # tensorboard visualization
    self.writer = SummaryWriter(comment="_bert_classifier")
    # model
    self.model = model
    self.device = "cuda" if torch.cuda.is_available() else "cpu"


  def train(self, train_data, val_data):
    train, val = SSTDataset(train_data), SSTDataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=self.batch_size,
                                                   shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=self.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    self.model.to(self.device)

    best_val_acc = 0.0
    best_val_loss = float('inf')

    for epoch_num in range(self.epochs):

      ### train ###
      self.model.train()
      total_acc_train = 0
      total_loss_train = 0

      for train_input, train_label in tqdm(train_dataloader):
        train_label = train_label.to(self.device)
        # (2, 1, 512) -> (2, 512)
        mask = train_input['attention_mask'].squeeze(1).to(self.device)
        input_id = train_input['input_ids'].squeeze(1).to(self.device)

        # input_id & mask shape: (batch_size, max_seq_len); e.g. (2, 512)
        output = self.model(input_id, mask)

        batch_loss = criterion(output, train_label.long())
        total_loss_train += batch_loss.item()

        acc = (output.argmax(dim=1) == train_label).sum().item()
        total_acc_train += acc

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()


      ### validation ###
      self.model.eval()
      total_acc_val = 0
      total_loss_val = 0

      with torch.no_grad():
        for val_input, val_label in val_dataloader:
          val_label = val_label.to(self.device)
          mask = val_input['attention_mask'].squeeze(1).to(self.device)
          input_id = val_input['input_ids'].squeeze(1).to(self.device)

          output = self.model(input_id, mask)

          batch_loss = criterion(output, val_label.long())
          total_loss_val += batch_loss.item()

          acc = (output.argmax(dim=1) == val_label).sum().item()
          total_acc_val += acc


      ### summary ###
      train_loss = total_loss_train / len(train_data)
      train_acc = total_acc_train / len(train_data)
      val_loss = total_loss_val / len(val_data)
      val_acc = total_acc_val / len(val_data)
      self.update_summary(epoch_num, train_loss, train_acc, val_loss, val_acc)

      ### save checkpoint ###
      if (val_acc > best_val_acc and val_loss < best_val_loss):
        self.save_checkpoint(epoch_num, val_acc, val_loss)
        best_val_acc = val_acc
        best_val_loss = val_loss

  def update_summary(self, epoch_num, train_loss, train_acc, val_loss, val_acc):
    print(f'Epochs: {epoch_num} \
      | Train Loss: {train_loss: .3f} \
      | Train Accuracy: {train_acc: .3f} \
      | Val Loss: {val_loss: .3f} \
      | Val Accuracy: {val_acc: .3f}')

    self.writer.add_scalar("loss/train", train_loss, epoch_num)
    self.writer.add_scalar("accuracy/train", train_acc, epoch_num)
    self.writer.add_scalar("loss/val", val_loss, epoch_num)
    self.writer.add_scalar("accuracy/val", val_acc, epoch_num)

  def save_checkpoint(self, epoch, acc, loss):
    out_path = "models/bert_classifier_epoch{}_acc{:.2f}_loss{:.2f}.pt".format(
      epoch, acc, loss)
    torch.save(self.model.state_dict(), out_path)
    print(f"Model saved to {out_path}")


def main():
  # args
  parser = argparse.ArgumentParser()
  parser.add_argument("--bs", type=int, default=16, help="batch size")
  parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
  parser.add_argument("--epochs", type=int, default=2, help="num of epochs")
  parser.add_argument("--datapath", type=str,
      default="/media/tianlu/SSD/datasets/stanford-sentiment-treebank/train.csv",
      help="path to training and validation data")
  parser.add_argument("--classes", type=int, default=2,
      help="number of classes in output layer")
  args = parser.parse_args()
  print(args)

  # hyperparameters
  batch_size = args.bs
  lr = args.lr
  epochs = args.epochs

  # data
  df = pd.read_csv(args.datapath)
  # 90%-10% split
  np.random.seed(112)
  df_train, df_val = np.split(df.sample(frac=1, random_state=42), [int(.9*len(df))])
  print(f"train:{len(df_train)}, val:{len(df_val)}")

  # model
  model = BertClassifier(dropout=0.5, embedding_dim=768, num_classes=args.classes)
  summary(model, [(2, 64), (2, 64)], dtypes=[torch.long, torch.long])

  # train
  trainer = Trainer(model, batch_size, lr, epochs)
  trainer.train(df_train, df_val)


if __name__ == "__main__":
  main()

