from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_texts = ["I will watch memento tonight", "let me think"]
bert_input = tokenizer(example_texts, padding='max_length', max_length=10,
  truncation=True, return_tensors="pt")

print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])

