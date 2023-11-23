import torch
from transformers import BertTokenizer

from bert_model import BertClassifier
from dataset import labels


tech_news = "tech helps disabled speed demons an organisation has been launched to encourage disabled people to get involved in all aspects of motorsport  which is now increasingly possible thanks to technological innovations."

sports_news = "The Texas Rangers defeated the Houston Astros 11-4 in Game 7 of the American League Championship Series Monday to win the American League pennant."

example_texts = [tech_news, sports_news]

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
