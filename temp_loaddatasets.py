from datasets import load_dataset_builder
from datasets import load_dataset
from transformers import AutoTokenizer
import torch


# Transformerモデル，データセットともに，論文などに表記する場合は権利をちゃんと調べて書く


# cudaが使用可能なら使用する
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rotten tomatoesという映画を見たレビューの文のロード
# ds_builder = load_dataset_builder("rotten_tomatoes")

# glueのうち，microsoft research paraphrase corpus(MRPC)を使用; 5801ペアのテキストに対して，それぞれの文のペアが同じ内容を指す文化どうかをラベルとして持っているようなデータセット
ds_builder = load_dataset_builder("glue", "mrpc")
print(ds_builder.info.description)
print(ds_builder.info.features)

# dataset = load_dataset("rotten_tomatoes", split="train")
# print(dataset[0]["text"])

dataset = load_dataset("glue", "mrpc", split="train")
print(f"データセット概要")
print(dataset)
print()
print(f"sentence1 example: {dataset[0]['sentence1']}")
print()

# トークン化
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenization(example):
    return tokenizer(example["sentence1"])

dataset = dataset.map(tokenization, batched=True)

# print(dataset[0]["input_ids"])

# PyTorchのために型変換
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"], device=device)
print(f"トークナイズ後(torch, device選択)")
print(dataset[0])
print(tokenizer.convert_ids_to_tokens(dataset[0]["input_ids"]))
print()

# これ使おうと思ってたけど，データの素性よくわからんから，せめてglueのデータセットだと，1文単位を確保できそう（それに有名なベンチマークの文を使える点も都合いいことありそう
# 現状，OpenSesamiの論文を参考に，そこで，関係代名詞を使用した文と，そうでない文を分けて取り出せるのならば，それは都合がよさそう：
# データについて詳細を見たところ，特定のルールに従って，データを作成していた．これもなしではないが，調査していた"and"ではなく，関係詞節になってしまうこと，照応関係を含まない場合文が短くなる可能性がある
# →逆に，照応関係を含む分を生成してそちらの文では収束しないことに言及してもよいかも

# 結局，glueのうちmnliのデータを使うのがよさそう





