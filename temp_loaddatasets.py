from datasets import load_dataset_builder
from datasets import load_dataset
from transformers import AutoTokenizer


ds_builder = load_dataset_builder("rotten_tomatoes")

print(ds_builder.info.description)
print(ds_builder.info.features)

dataset = load_dataset("rotten_tomatoes", split="train")

print(dataset)
print(dataset[0])
print(dataset[0]["text"])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenization(example):
    return tokenizer(example["text"])

dataset = dataset.map(tokenization, batched=True)

print(dataset[0]["input_ids"])
print(tokenizer.convert_ids_to_tokens(dataset[0]["input_ids"]))
