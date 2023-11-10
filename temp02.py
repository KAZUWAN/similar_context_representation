from transformers import BertTokenizer, BertModel
import torch


# BERTモデルとトークナイザのロード
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 入力テキストの準備
input_text = "Your input text here."
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# BERTモデルの出力を取得
outputs = model(**inputs, output_hidden_states= True, output_attentions= True)

# BERTのアテンション層を取得
attention_layer = model.encoder.layer[11].attention

# self.queryを取得
query_weights = attention_layer.self.query.weight

print("Shape of self.query weights:", query_weights.shape)

# print(model.encoder.layer)

print(f'outputs keys: {list(outputs.keys())}')
print(f"outputs hidden states length: {len(outputs['hidden_states'])}")
print(f"output hidden states 1 shape: {outputs['hidden_states'][1].shape}")

input_layer2 = outputs['hidden_states'][1]
mixed_query = attention_layer.self.query(input_layer2)
# print(f'input_layer2: {input_layer2}')
print(f'mixed query shape: {mixed_query.shape}')
print(f'mixed query: {mixed_query}')

new_x_shape = mixed_query.size()[:-1] + (attention_layer.self.num_attention_heads, attention_layer.self.attention_head_size)
x = mixed_query.view(*new_x_shape)
query_layer = x.permute(0, 2, 1, 3)

print(f'config hidden size: {model.config.hidden_size}')
print(f'query layer shape: {query_layer.shape}')
# print(f'query layer: {query_layer}')
print(f'model config: \n{model.config}')
# print(f'model: {model}')
bertselfoutput_layer = attention_layer.output
selfoutput_weights = bertselfoutput_layer.dense.weight
print(f'self output weight: {selfoutput_weights}')
print(f'self output weight shape: {selfoutput_weights.shape}')


# 10層目の出力（hidden_states)
output_10 = outputs['hidden_states'][9]
print(f'output_10 shape: {output_10.shape}')
# 11層目のqueryの線形層に入れる
attention_layer11 = model.encoder.layer[10].attention
query = attention_layer11.self.query(output_10)

print(f'query: {query}')

# 手動で取り出した11層目の線形層の重みと，隠れ層との掛け算で結果が同じか

mixed_query = torch.matmul(output_10, attention_layer11.self.query.weight) + attention_layer11.self.query.bias
print(f'mixed_query shape: {mixed_query.shape}')
print(f'miexd_query: \n{mixed_query}')

'''こちらの計算式と，線形層の計算が一致．すなわち，取り出される重み行列は，これを転置して行列積をとっている（なので，線形層の重みを取り出すときは注意が必要'''
mixed_query = torch.matmul(output_10, torch.transpose(attention_layer11.self.query.weight, 0, 1)) + attention_layer11.self.query.bias 
print(f'mixed_query shape: {mixed_query.shape}')
print(f'miexd_query: \n{mixed_query}')


key = attention_layer11.self.key(output_10)
print(f'key: {key}')

mixed_query = torch.matmul(output_10, torch.transpose(attention_layer11.self.key.weight, 0, 1)) + attention_layer11.self.key.bias 
print(f'output_10[0,0].shape:{output_10[0,0].shape}')
print(f'output_10[0,0]:\n{output_10[0,0]}')
print(f'attention_layer11.self.key.bias.shape:{attention_layer11.self.key.bias.shape}')
print(f'attention_layer11.self.key.bias:\n{attention_layer11.self.key.bias}')
print(f'mixed_query shape: {mixed_query.shape}')
print(f'miexd_query: \n{mixed_query}')

# 以下の結果より，matmulは想定通りの行列計算をしてくれている
hoge = torch.tensor([1,2,3])
print(f'hoge: {hoge}')
hoge2 = torch.tensor([[1,1,1],[2,2,2],[3,3,3]])
print(f'hoge2.shape {hoge2.shape}')
print(f'hoge2.sum(dim=0)', hoge2.sum(dim=0))
print(f'hoge2[0:3]: {hoge2[0:3]}')
print(f"hoge2: {hoge2}")
hoge3 = torch.matmul(hoge, hoge2)
print(f'hoge3 = torch.matmul(hoge, hoge2):\n {hoge3}')
hoge4 = torch.matmul(hoge, torch.transpose(hoge2, 0, 1))
print(f'hoge4 = torch.matmul(hoge, torch.transpose(hoge2, 0, 1)):\n {hoge4}')

print(f'shape hoge: {hoge.shape}')
print(f'shape torch.transpose(hoge2, 0, 1): {torch.transpose(hoge2, 0, 1).shape}')












