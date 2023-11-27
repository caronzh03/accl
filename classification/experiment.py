import torch
from transformers import BertTokenizer

from bert_model import BertClassifier


pos_review = "a gripping movie , played with performances that are all understated and touching."
neg_review = "a valueless kiddie paean to pro basketball underwritten by the nba."

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
bert_classifier = bert_classifier.to("cuda")
bert_classifier.eval()

input_ids = bert_input["input_ids"].squeeze(1).to("cuda")
mask = bert_input["attention_mask"].squeeze(1).to("cuda")
output = bert_classifier(input_ids=input_ids, mask=mask)
print(output)
print(output.argmax(dim=1))
