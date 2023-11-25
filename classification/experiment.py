import torch
from transformers import BertTokenizer

from bert_model import BertClassifier


pos_review = "this movie told a very touch story and I loved it!"
neg_review = "the movie was confusing to watch given the reverse-chronilogical order of events"

example_texts = [pos_review, neg_review]

# tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_input = tokenizer(example_texts, padding='max_length', max_length=512,
  truncation=True, return_tensors="pt")
print(f"bert_input: {bert_input}")

# inference
bert_classifier = BertClassifier()
model_path = "models/best_bert_classifier.pt"
bert_classifier.load_state_dict(torch.load(model_path))
bert_classifier.eval()

output = bert_classifier(input_ids=bert_input['input_ids'], mask=bert_input['attention_mask'])
print(output)
print(output.argmax(dim=1))
