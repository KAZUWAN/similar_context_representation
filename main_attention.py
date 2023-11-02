
import os
import torch
import copy
import numpy as np
import datetime

from transformers import BertTokenizer, BertModel
from src import config
from src.util import load_data
from src.util import make_datacollection02
import visualize_attention.mean_attentionw
import visualize_attention.attention_visualize02
import visualize_attention.for_attention_func01
from src.util import add_special_token
import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    input_path_tr = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), config.DATA_TRAIN_PATH))
    train_dataset = load_data.load_json_data(input_path_tr)

    creater = make_datacollection02.CreateOriginal(train_dataset)

    # creater.remove_only_o(remov= True)
    # creater.create_sentence_list()
    # creater.create_sentences_dict()
    # all_sentences = creater.return_something()
    # print(f"length all sentences : {len(all_sentences['sentence_list'])}")
    # print(all_sentences['sentence_list'][0])
    # print(all_sentences['sentence_list'][2])
    # print(all_sentences['sentence_list'][4])

    sentences_list = copy.copy(creater.ori_sentence_list)
    print(f"発話文数：{len(sentences_list)}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    # visualize_attention.mean_attentionw.mean_attention_w_by_hiddenlayer(sentences_list, bert_model= bert_model, tokenizer= tokenizer)

    length_list = [len(length) for length in sentences_list]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    # print(list(range(1, 30, 1)))
    bin = [i+0.5 for i in range(29)]
    plt.title(f"sentence length", fontsize=16)
    ax.set_xlabel("word count", fontsize=14)
    ax.set_ylabel("frequency", fontsize=14)
    ax.set_xticks(np.arange(1, 29, 1), fontsize=12)
    ax.hist(length_list, bins= bin, ec= "black")

    # save histogram
    filepath = os.path.dirname(os.path.abspath(__file__))
    now = datetime.datetime.now()
    save_file = f'visualize_attention/figures/histo_sentence_lenght_{now.year:>04}{now.month:>02}{now.day:>02}{now.hour:>02}{now.minute:>02}{now.second:>02}.png'
    filepath = os.path.join(filepath, save_file)
    # plt.savefig(save_file)
    # plt.show()
    plt.close()



    # print(list(set(length_list)))
    # for i in range(200):
    #     print(f"{i}: {sentences_list[i]}")
    # assert False


    # 調べたい文を手動で，文のリストの後ろに追加する
    # probe_sentence1 = ["hi", ",", "i", "want", "to", "attack", "a", "restaurant", "reservation", "."]
    # sentences_list.append(probe_sentence1)
    probe_sentence2 = ["the", "lord", "that", "can", "hurt", "the", "prince", "could", "comfort", "wizard", "by", "himself"]
    sentences_list.append(probe_sentence2)
    # # Open Sesamiの論文のPDF 7ページの式4の下の文をを参照↓
    # probe_sentence3 = ["verbs", "agree", "with", "a", "single", "subject", ",", "and", "anaphor", "take", "a", "single", "noun", "phrase", "as", "their", "antecedent", "."]
    # sentences_list.append(probe_sentence3)
    # probe_sentence4 = ["verbs", "agree", "with", "a", "single", "subject", ",", "and", "i", "want", "to", "make", "a", "restaurant", "reservation", "."]
    # sentences_list.append(probe_sentence4)
    max_length = max(length_list)
    
    sentences_add_token_dic = add_special_token.add_specialtokens(sentences_list, max_length, plus_token= True, padding= False)

    sentences_add_token = sentences_add_token_dic['specialtokens_added']
    attention_mask_list = sentences_add_token_dic['attention_mask_list']
    input_ids_list = [tokenizer.convert_tokens_to_ids(k) for k in sentences_add_token]


    # sentence_number_list = [0, 17, 60, 86, 96, 124, -4, -3, -2, -1] # any number
    sentence_number_list = [-1]
    for sentence_number in sentence_number_list:
        print(f"{sentence_number}")
        input_ids = torch.tensor(input_ids_list[sentence_number]).unsqueeze(dim= 0)
        
        bert_out = bert_model(input_ids, output_hidden_states= True, output_attentions= True)
        attentions = bert_out['attentions']
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(dim= 0))
        for i, attention_n in enumerate(attentions):
            attention_w = attention_n[0]
            # visualize_attention.attention_visualize02.show_attention_heatmap(attention_w_tensor= attention_w, tokens= tokens, layer_n=i, save = True, show= False)
            visualize_attention.for_attention_func01.pileup_attention(attention_w_tensor= attention_w, tokens= tokens, layer_n=i, save = True, show= False)

            # 各レイヤーでどの単語がどの単語の情報を受け取る回数が多いか出力する．CLS, SEPトークンを除いて，どれに注意しているか見ることもできる
            visualize_attention.for_attention_func01.most_attendingto_wordfreq(attention_w_tensor= attention_w, frag_mask= True, tokens= tokens, layer_n=i, save = True, show= False)


    # num_attention_heads = bert_model.config.num_attention_heads
    # print(f'num_attention_heads: {num_attention_heads}')
    # print(f'bert_model attention head size: {int(bert_model.config.hidden_size / num_attention_heads)}')
    # attention_values = bert_out['last_hidden_state']
    # print(f'attention values shape: {attention_values.shape}')
    # attention_tensors = attention_values.chunk(num_attention_heads, dim=2)

    # for i, tensor in enumerate(attention_tensors):
    #     print(f'Head {i+1} value shape: {tensor.shape}')

    # print(f'{bert_model.value}')


    