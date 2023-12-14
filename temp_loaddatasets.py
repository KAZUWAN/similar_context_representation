from datasets import load_dataset_builder
from datasets import load_dataset
from transformers import AutoTokenizer
import torch


# cudaが使用可能なら使用する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rotten tomatoesという映画を見たレビューの文のロード
ds_builder = load_dataset_builder("rotten_tomatoes")

print(ds_builder.info.description)
print(ds_builder.info.features)

dataset = load_dataset("rotten_tomatoes", split="train")

print(f"データセット概要")
print(dataset)
print()
# print(dataset[0])
print(dataset[0]["text"])
print(dataset[1]["text"])
print(dataset[2]["text"])
print()

# ここで，トークナイズする前に，文単位でデータを区切りたい
# たまにj.r.r.のような人の名前？のために使用するピリオドを考慮しなければならない
# さらには，かっこやコロンのような記号を持つものを今回は取り除くのが良いと思う
# ピリオドが二つ以上ある文がいくつあるか調べる
# ↑すなわち，データの素性調査

# とりあえずピリオドが一つ以下の文だけを対象にする


# トークン化
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenization(example):
    return tokenizer(example["text"])

dataset = dataset.map(tokenization, batched=True)

# print(dataset[0]["input_ids"])

# PyTorchのために型変換
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"], device=device)
print(f"トークナイズ後(torch, device選択)")
print(dataset[0])
print(tokenizer.convert_ids_to_tokens(dataset[0]["input_ids"]))
print()




