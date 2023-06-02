from transformers import BertTokenizer

# bert tokenizerでトークンIDにする
def bert_ids(data_tokens):
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    input_id=[tokenizer.convert_tokens_to_ids(k) for k in data_tokens]
    return {"input_id":input_id}