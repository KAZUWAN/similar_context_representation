import json
import numpy as np
from keras.utils import np_utils
from transformers import BertTokenizer
import torch


# jsonのデータ読み込み
# データトークンとラベル（とラベル番号）を取得
def json_loading(path):
    json_open=open(path,"r")
    json_load=json.load(json_open)
    json_open.close()
    
    data_tokens=[]
    data_labels=[]
    data_labels_num=[]
    # max_length=0
    
    for i in range(len(json_load)):
        for j in range(len(json_load[i]["turns"])):

            temp_tokens=json_load[i]["turns"][j]["user_utterance"]["tokens"]#ユーザー発話のトークンを獲得．リスト等に追加する必要がある

            temp_labels=["o"]*len(json_load[i]["turns"][j]["user_utterance"]["tokens"]) #ペアにラベルを振る必要がある．# 逆に0この時は，全てに”o”を振る必要がある
            temp_labels_num=[0]*len(json_load[i]["turns"][j]["user_utterance"]["tokens"]) #ペアにラベルを振る必要がある．# 逆に0この時は，全てに”o”を振る必要がある

            if len(json_load[i]["turns"][j]["user_utterance"]["slots"])>=1:# よって，同じ長さの”o”で埋められたリストを作成し，1単語目のスロットバリューの位置を”Ｂ”，2単語目以降を”I”ラベルに置き換えることで# ラベル付きのデータセットを作成できそう
                for l in range(len(json_load[i]["turns"][j]["user_utterance"]["slots"])):# slotsが複数ある場合のために繰り返す

                    temp_labels[json_load[i]["turns"][j]["user_utterance"]["slots"][l]["start"]]="B"#スロットバリューの位置を取得，置き換え
                    temp_labels_num[json_load[i]["turns"][j]["user_utterance"]["slots"][l]["start"]]=1#スロットバリューの位置を取得，置き換え

                    temp_labels[json_load[i]["turns"][j]["user_utterance"]["slots"][l]["start"]+1:json_load[i]["turns"][j]["user_utterance"]["slots"][l]["exclusive_end"]]=["I"]*(json_load[i]["turns"][j]["user_utterance"]["slots"][l]["exclusive_end"]-json_load[i]["turns"][j]["user_utterance"]["slots"][l]["start"]-1)#スロットバリューの位置を取得，置き換え
                    temp_labels_num[json_load[i]["turns"][j]["user_utterance"]["slots"][l]["start"]+1:json_load[i]["turns"][j]["user_utterance"]["slots"][l]["exclusive_end"]]=[2]*(json_load[i]["turns"][j]["user_utterance"]["slots"][l]["exclusive_end"]-json_load[i]["turns"][j]["user_utterance"]["slots"][l]["start"]-1)#スロットバリューの位置を取得，置き換え
            data_tokens.append(temp_tokens)
            data_labels.append(temp_labels)
            data_labels_num.append(temp_labels_num)
            # if len(temp_tokens)>max_length:
            #     max_length=len(temp_tokens)
            
    # return {"tokens":data_tokens,"labels":data_labels,"labels_num":data_labels_num,"max_length":max_length}
    return {"tokens":data_tokens,"labels":data_labels,"labels_num":data_labels_num}


def one_hot(data_labels_num):
    one_hot_labels=[]

    for i in range(len(data_labels_num)):
        one_hot_labels.append(np_utils.to_categorical(data_labels_num[i],3))

    one_hot_labels=np.array(one_hot_labels)
    one_hot_labels_arr=np.copy(one_hot_labels)
    
    return {"one_hot_":one_hot_labels_arr}
            

# bert分散表現用に，トークンの前後に[CLS][SEP],ラベル列に"o","0"をつける
# 発話文の一番大きいサイズに合わせてpadding
def token_add(data_tokens,data_labels,data_labels_num):
    max_length=max([len(i) for i in data_tokens])
    attention_mask=[]
    for i in range(len(data_tokens)):
        data_tokens[i].append("[SEP]")
        data_labels[i].append("o")
        data_labels_num[i].append(0)
        data_tokens[i].insert(0,"[CLS]")
        data_labels[i].insert(0,"o")
        data_labels_num[i].insert(0,0)
        attention_mask.append([1]*len(data_tokens[i]))
        if len(data_tokens[i])<max_length+2:
            temp_pad=["[PAD]"]*(max_length+2-len(data_tokens[i]))
            temp_o=["o"]*(max_length+2-len(data_tokens[i]))
            temp_0=[0]*(max_length+2-len(data_tokens[i]))

            data_tokens[i].extend(temp_pad)
            data_labels[i].extend(temp_o)
            data_labels_num[i].extend(temp_0)
            attention_mask[i].extend(temp_0)


    return {"tokens_add":data_tokens,"labels_add":data_labels,"labels_num_add":data_labels_num,"attention_mask":attention_mask}
            


# bert tokenizerでトークンIDにする
def bert_ids(data_tokens):
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    input_id=[tokenizer.convert_tokens_to_ids(k) for k in data_tokens]
    return {"input_id":input_id}


# 読み込み，トークン，ラベル配列獲得
def data_list(path):
    data=json_loading(path)
    # print(datakeys())
    tokens=data["tokens"]
    labels=data["labels"]
    labels_num=data["labels_num"]
    max_length=data["max_length"]

    # bert用に文の前後にラベル付与
    data_add=token_add(tokens,labels,labels_num,max_length)
    # print(data_train_add.keys())
    tokens_add=data_add["tokens_add"]
    labels_add=data_add["labels_add"]
    labels_num_add=data_add["labels_num_add"]
    attention_mask=data_add["attention_mask"]

    # ラベル番号をone-hotに
    one_hot_labels=one_hot(labels_num_add)
    one_hot_labels_arr=one_hot_labels["one_hot_"]

    # bert_tokenizer でトークンID化 
    bert_input_id=bert_ids(tokens_add)
    input_id=bert_input_id["input_id"]

    return({"data":data,"tokens":tokens,"labels":labels,"label_n":labels_num,"tokens_add":tokens_add,"labels_add":labels_add\
        ,"attention_mask":attention_mask,"labels_num_add":labels_num_add,"one_hot_":one_hot_labels_arr,"input_id":input_id})


# gain word_emb
def Bert_CRF(input_id,attention_mask,model):

    tokens_tensor=torch.tensor([input_id]) 
    attention_tensor=torch.tensor([attention_mask])

    hidden_states=model(tokens_tensor,attention_tensor,output_hidden_states=True)
    
    return hidden_states 