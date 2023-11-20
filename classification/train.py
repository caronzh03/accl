import torch
from torch import nn
from torch.optim import Adam
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import numpy as np

from dataset import Dataset
from bert_model import BertClassifier


def train(model, train_data, val_data, learning_rate, epochs):
  # initialize tensorboard visualization
  writer = SummaryWriter(comment="_bert_classifier")

  train, val = Dataset(train_data), Dataset(val_data)
  train_dataloader = torch.utils.data.DataLoader(train, batch_size=2,
                                                 shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=learning_rate)

  model.to(device)
  criterion.to(device)

  best_val_acc = 0.0
  best_val_loss = float('inf')

  for epoch_num in range(epochs):

    ### train ###
    model.train()
    total_acc_train = 0
    total_loss_train = 0

    for train_input, train_label in tqdm(train_dataloader):
      train_label = train_label.to(device)
      mask = train_input['attention_mask'].to(device)
      # TODO: check dim
      input_id = train_input['input_ids'].squeeze(1).to(device)
      
      output = model(input_id, mask)

      batch_loss = criterion(output, train_label.long())
      # TODO: check batch_loss.item() output
      total_loss_train += batch_loss.item()
      
      # TODO: check model output: should be a logits vector
      acc = (output.argmax(dim=1) == train_label).sum().item()
      total_acc_train += acc
      
      model.zero_grad()
      batch_loss.backward()
      optimizer.step()


    ### validation ###
    model.eval()
    total_acc_val = 0
    total_loss_val = 0

    with torch.no_grad():
      for val_input, val_label in val_dataloader:
        val_label = val_label.to(device)
        mask = val_input['attention_mask'].to(device)
        input_id = val_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        batch_loss = criterion(output, val_label.long())
        total_loss_val += batch_loss.item()

        acc = (output.argmax(dim=1) == val_label).sum().item()
        total_acc_val += acc


    ### epoch summary ###
    train_loss = total_loss_train / len(train_data)
    train_acc = total_acc_train / len(train_data)
    val_loss = total_loss_val / len(val_data)
    val_acc = total_acc_val / len(val_data)
    print(f'Epochs: {epoch_num} \
      | Train Loss: {train_loss: .3f} \
      | Train Accuracy: {train_acc: .3f} \
      | Val Loss: {val_loss: .3f} \
      | Val Accuracy: {val_acc: .3f}')

    writer.add_scalar("loss/train", train_loss, epoch_num)
    writer.add_scalar("accuracy/train", train_acc, epoch_num)
    writer.add_scalar("loss/val", val_loss, epoch_num)
    writer.add_scalar("accuracy/val", val_acc, epoch_num)

    ### save checkpoint ###
    if (val_acc > best_val_acc and val_loss < best_val_loss):
      save_checkpoint(model, epoch_num, val_acc, val_loss)
      best_val_acc = val_acc
      best_val_loss = val_loss
  

def save_checkpoint(model, epoch, acc, loss):
  out_path = "models/bert_classifier_epoch{}_acc{:.2f}_loss{:.2f}.pt".format(
    epoch, acc, loss)
  torch.save(model.state_dict(), out_path)
  print(f"Model saved to {out_path}")


def test(model, test_data):
  test = Dataset(test_data)
  test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  total_acc_test = 0
  model.eval()
  with torch.no_grad():
    for test_input, test_label in test_dataloader:
      test_label = test_label.to(device)
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)

      output = model(input_id, mask)

      acc = (output.argmax(dim=1) == test_label).sum().item()
      total_acc_test += acc
  
  print(f"Test accuracy: {total_acc_test / len(test_data): .3f}")



def main():

  # hyperparameters
  epochs = 50
  lr = 1e-6

  # data
  datapath = '/media/tianlu/SSD/datasets/bbc_classification/bbc-text.csv'
  df = pd.read_csv(datapath)

  np.random.seed(112)
  # 80% - 10% - 10% split
  df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                       [int(.8*len(df)), int(.9*len(df))])
  print(f"train:{len(df_train)}, val:{len(df_val)}, test:{len(df_test)}")

  # model
  model = BertClassifier()
  summary(model, [(2, 64), (2, 64)], dtypes=[torch.long, torch.long])

  # train
  train(model, df_train, df_val, lr, epochs)

  # test
  test(model, df_test)



if __name__ == "__main__":
  main()

