import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer


def tokenize(tokenizer, examples):
  return tokenizer(examples['sentence'], truncation=True, padding='max_length')


def main():
  # load training data, tokenizer, pretrained model and metrics
  dataset = load_dataset('glue', 'sst2')
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
  model = BertForSequenceClassification.from_pretrained('bert-base-cased', return_dict=True)
  metric = load_metric('glue', 'sst2')

  dataset = dataset.map(lambda examples: tokenize(tokenizer, examples), batched=True)
  encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

  # change to pytorch accepted format
  columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
  encoded_dataset.set_format(type='torch', columns=columns)

  def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)

  # training parameters
  args = TrainingArguments(
    "ft-sst2", # output path
    evaluation_strategy="epoch", # evaluate after each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2
  )

  # trainer
  trainer = Trainer(
    model, 
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
  )

  trainer.train()

  # evaluate on validation set
  trainer.evaluate()


if __name__ == "__main__":
  main()
