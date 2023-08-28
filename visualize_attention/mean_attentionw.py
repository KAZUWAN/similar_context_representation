
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.util import add_special_token


def mean_attention_w_by_hiddenlayer(sentences: list, bert_model, tokenizer) -> None:

    length_list = [len(length) for length in sentences]
    max_length = max(length_list)
    print(f'max length: {max_length}')

    sentences_add_token_dic = add_special_token.add_specialtokens(sentences, max_length, plus_token= True, padding= False)

    sentences_add_token = sentences_add_token_dic['specialtokens_added']
    attention_mask_list = sentences_add_token_dic['attention_mask_list']
    input_ids_list = [tokenizer.convert_tokens_to_ids(k) for k in sentences_add_token]

    print(f'length sentences after add token: {len(sentences_add_token)}')
    print(f'length attention masks: {len(attention_mask_list)}')
    print(f'length input ids list: {len(input_ids_list)}')
    print(f'sentence: {sentences_add_token[0]}')
    print(f'attention_mask: {attention_mask_list[0]}')
    print(f'input_ids: {input_ids_list[0]}')
    print(f'ids to tokens: {tokenizer.convert_ids_to_tokens(input_ids_list[0])}')
    # attention w を抽出後，[sep]トークン以外の単語にどの程度着目するのか層を重ねるごとにどう変化するか，文を問わず平均して出力してみる
    # attention w 獲得

    store_mean_attention = np.zeros((len(input_ids_list), 12))
    for i in range(len(input_ids_list)):
        mean_by_layer = np.zeros(12)
        input_ids = torch.tensor(input_ids_list[i]).unsqueeze(dim= 0)
        attention_mask = torch.tensor(attention_mask_list[i]).unsqueeze(dim= 0)
        bert_output = bert_model(input_ids = input_ids, attention_mask= attention_mask, output_hidden_states= False, output_attentions= True)
        attentions = bert_output['attentions']
        
        for j, attention_layer in enumerate(attentions):
            layer_n = j
            attention_tensor = attention_layer[0] # 0 is batch number
            attentionw = attention_tensor[j]
            attentionw = attentionw[:-1, :-1]  # sep 以外の抽出
            attention_mean = attentionw.mean().detach().numpy()
            mean_by_layer[j] = attention_mean
            # print(mean_by_layer)
        # np.append(store_mean_attention, mean_by_layer, axis= 0)
        store_mean_attention[i] = mean_by_layer
        # print(f'store: {store_mean_attention[0:3]}')
        # if i == 2:
        #     print(f'store mean attention: {store_mean_attention[0:5]}')
        #     print(f'column 0: {store_mean_attention[:, 0]}')
        #     print(f'column 0 mean: {store_mean_attention[:, 0].mean()}')
        #     print(f'column 0 mean: {store_mean_attention[:, 1].mean()}')
        #     print(f'column 0 mean: {store_mean_attention[:, 2].mean()}')
        #     print(f'column 0 mean: {store_mean_attention[:, 3].mean()}')
        #     print(f'column 0 mean: {store_mean_attention[:, 4].mean()}')
        #     break

    mean_all_attention = []
    for k in range(len(store_mean_attention[0])):
        mean_all_attention.append(store_mean_attention[:, k].mean())
    print(mean_all_attention)
    fig = plt.figure()
    plt.plot(list(range(len(mean_all_attention))), mean_all_attention)
    plt.show()
    



