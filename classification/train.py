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
from eval import Evaler


class State(object):
  def __init__(self, name, summary_writer):
    """
    Store training and validation internal states.
    """
    # train or validate
    self.name = name
    # total accuracy count
    self.total_acc = 0
    # total loss count
    self.total_loss = 0
    # total number of samples processed
    self.n_processed = 0
    # number of batches processed
    self.step = 0
    # tensorboard
    self.writer = summary_writer
    # tqdm progress, set during epoch training/validation
    self.progress_bar = None

  def set_progress_bar(self, bar):
    self.progress_bar = bar

  def update_metrics(self, epoch, batch_loss,
                     batch_accuracy, batch_size) -> (float, float):
    """
    Updates loss and accuracy on tensorboard graph and tqdm progress bar.
    Called per step.

    @Return running loss, running accuracy.
    """
    self.total_loss += batch_loss * batch_size
    self.total_acc += batch_accuracy
    self.n_processed += batch_size
    self.step += 1

    loss = self.total_loss / self.n_processed
    accuracy = self.total_acc / self.n_processed

    self.progress_bar.set_description("Epoch {} ({}): loss={:.3f}, accuracy={:.3f}".format(
        epoch, self.name, loss, accuracy))

    if self.step % 100 == 0:
      self.writer.add_scalar("loss/" + self.name, loss, self.step)
      self.writer.add_scalar("accuracy/" + self.name, accuracy, self.step)

    return loss, accuracy



class Trainer(object):
  def __init__(self, model, batch_size, learning_rate, epochs):
    """
    Executes training and validation loops.
    """
    # hyperparameters
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.epochs = epochs
    # model
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = model.to(self.device)
    # loss function
    self.criterion = nn.CrossEntropyLoss().to(self.device)
    # optimizer
    self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
    # metrics
    self.writer = SummaryWriter(comment="_bert_classifier")
    self.train_state = State("train", self.writer)
    self.val_state = State("validate", self.writer)

  def train(self, train_data, val_data):
    """
    Train model.
    """
    train, val = SSTDataset(train_data), SSTDataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=self.batch_size,
                                                   shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=self.batch_size)

    for epoch_num in range(self.epochs):
      self.train_epoch(train_dataloader, epoch_num)
      loss, accuracy = self.validate_epoch(val_dataloader, epoch_num)
      # save model snapshot per epoch
      self.save_checkpoint(epoch_num, loss, accuracy)

  def train_epoch(self, train_dataloader, epoch):
    """
    Train an epoch.
    """
    self.model.train()
    train_progress = tqdm(train_dataloader, ncols=150)
    self.train_state.set_progress_bar(train_progress)
    for train_input, train_label in train_progress:
      train_label = train_label.to(self.device)
      # (2, 1, 512) -> (2, 512)
      mask = train_input['attention_mask'].squeeze(1).to(self.device)
      input_id = train_input['input_ids'].squeeze(1).to(self.device)

      # forward pass
      # input_id & mask shape: (batch_size, max_seq_len); e.g. (2, 512)
      output = self.model(input_id, mask)
      batch_loss = self.criterion(output, train_label.long())

      # backward propagation
      self.optimizer.zero_grad()
      batch_loss.backward()
      self.optimizer.step()

      # update metrics
      batch_acc = (output.argmax(dim=1) == train_label).sum().item()
      batch_size = output.shape[0]
      self.train_state.update_metrics(epoch, batch_loss.item(), batch_acc, batch_size)

  def validate_epoch(self, val_dataloader, epoch) -> (float, float):
    """
    Validates an epoch.
    Returns running validation loss & accuracy pair.
    """
    self.model.eval()
    val_loss, val_acc = None, None

    with torch.no_grad():
      validation_progress = tqdm(val_dataloader, ncols=150)
      self.val_state.set_progress_bar(validation_progress)
      for val_input, val_label in validation_progress:
        val_label = val_label.to(self.device)
        mask = val_input['attention_mask'].squeeze(1).to(self.device)
        input_id = val_input['input_ids'].squeeze(1).to(self.device)

        # forward pass
        output = self.model(input_id, mask)

        # update metrics
        batch_loss = self.criterion(output, val_label.long()).item()
        batch_acc = (output.argmax(dim=1) == val_label).sum().item()
        batch_size = output.shape[0]
        val_loss, val_acc = self.val_state.update_metrics(
            epoch, batch_loss, batch_acc, batch_size)

    return val_loss, val_acc

  def save_checkpoint(self, epoch, loss, acc):
    """
    Saves current model checkpoint. Called per epoch.
    """
    out_path = "models/bert_classifier_epoch{}_acc{:.3f}_loss{:.3f}.pt".format(
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
  # 85%-10%-5% train-val-eval
  np.random.seed(112)
  df_train, df_val, df_eval = np.split(df.sample(frac=1, random_state=42),
                                       [int(.85*len(df)), int(.95*(len(df)))])
  print(f"train:{len(df_train)}, val:{len(df_val)}, eval:{len(df_eval)}")

  # model
  model = BertClassifier(dropout=0.5, embedding_dim=768, num_classes=args.classes)
  summary(model, [(2, 64), (2, 64)], dtypes=[torch.long, torch.long])

  # train
  trainer = Trainer(model, batch_size, lr, epochs)
  trainer.train(df_train, df_val)

  # eval
  evaler = Evaler(model, batch_size)
  evaler.eval(df_eval)


if __name__ == "__main__":
  main()

