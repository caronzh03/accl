from datasets import load_dataset
import torch
import textbrewer
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
from transformers import BertTokenizerFast, BertForSequenceClassification, DistilBertForSequenceClassification


def encode(tokenizer, examples):
  return tokenizer(examples['sentence'], truncation=True, padding='max_length')


def simple_adaptor(batch, model_outputs):
  return {'logits': model_outputs[1]}


def main():
  # use glue sst2 as training data
  dataset = load_dataset('glue', 'sst2', split='train')
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
  dataset = dataset.map(lambda examples: encode(tokenizer, examples), batched=True)
  encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
  columns = ['input_ids', 'attention_mask', 'labels']
  encoded_dataset.set_format(type='torch', columns=columns)

  def collate_fn(examples):
    return dict(tokenizer.pad(examples, return_tensors='pt'))
  
  dataloader = torch.utils.data.DataLoader(encoded_dataset, collate_fn=collate_fn, batch_size=8)

  # define teacher & student models
  teacher_model = BertForSequenceClassification.from_pretrained('bert-base-cased')
  student_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')

  # print teacher & student models' parameters
  print("teacher model parameters:\n")
  result, _ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
  print(result)

  print("student model parameters:\n")
  result, _ = textbrewer.utils.display_parameters(student_model, max_level=3)
  print(result)

  # optimizer
  optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  teacher_model.to(device)
  student_model.to(device)

  # distillation configs
  train_config = TrainingConfig(device=device)
  distill_config = DistillationConfig()
  
  # distiller
  distiller = GeneralDistiller(train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model,
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

  # start distillation
  with distiller:
    # by default, distilled model will be saved to saved_model/
    distiller.train(optimizer, dataloader, scheduler_class=None, scheduler_args=None,
      num_epochs=1, callback=None)


if __name__ == "__main__":
  main()

