import os
import numpy as np
from transformers import BertTokenizer,BertModel
import torch
from . import Add_tokens # when using this program for module
from . import One_hot
from . import check_progress as prog
from . import Dataset_SIM_loading as load
from ..config  import *
# import Add_tokens # when using this program for execution
# import One_hot



class BertFunction():
    def __init__(self) -> None:
        self.tokenizer=BertTokenizer.from_pretrained(pretrained_name)
        self.model=BertModel.from_pretrained(pretrained_name)

    # input =(["tokens"],["labels"],["labels_number"])
    def get_data_list(self,data, to_ids = True):
        tokens=data[0]
        labels=data[1]
        labels_num=data[2]
        slots = data[3]

        # add tokens for BERT tokenizer
        data_add=Add_tokens.add_tokens(tokens,labels,labels_num, slots)
        tokens_add=data_add["tokens_add"]
        labels_add=data_add["labels_add"]
        labels_num_add=data_add["labels_num_add"]
        slots_add = data_add['slots_add']
        attention_mask=data_add["attention_mask"]

        # convert class label to one-hot
        one_hot_labels=One_hot.one_hot(labels_num_add)
        one_hot_labels_arr=one_hot_labels["one_hot_"]

        # convert tokens wiht Bert_tokenizer  
        if to_ids:
            bert_input_id=self.bert_ids(tokens_add)
            input_id=bert_input_id["input_id"]

            return({"tokens":tokens,"labels":labels, "label_n":labels_num, 'slots':slots, "tokens_add":tokens_add, "labels_add":labels_add, 'slots_add':slots_add\
            ,"attention_mask":attention_mask,"labels_num_add":labels_num_add,"one_hot_":one_hot_labels_arr,"input_id":input_id})

        else:
            return({"tokens":tokens,"labels":labels,"label_n":labels_num, 'slots':slots,"tokens_add":tokens_add,"labels_add":labels_add, 'slots_add':slots_add\
            ,"attention_mask":attention_mask,"labels_num_add":labels_num_add,"one_hot_":one_hot_labels_arr})


    def split_data(self,input_datapath, BATCH=BATCH_SIZE, to_ids = True):

        path=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),input_datapath))

        data_original=load.json_loading(path)

        tokens=data_original["tokens"]
        labels=data_original["labels"]
        labels_num=data_original["labels_num"]
        slots = data_original["slots"]

        # sort data
        length=[len(tokens[i]) for i in range(len(tokens))]
        sort_index=np.argsort(length)
        tokens_sort=[tokens[sort_index[i]] for i in range(len(sort_index))]
        labels_sort=[labels[sort_index[i]] for i in range(len(sort_index))]
        labels_num_sort=[labels_num[sort_index[i]] for i in range(len(sort_index))]
        slots_sort = [slots[sort_index[i]] for i in range(len(sort_index))]

        # split data by BATCH_SIZE
        data_split=[]

        if len(tokens)%BATCH == 0:
            r = 0
        else:
            r = 1
            
        for i in range((len(tokens)//BATCH)+r):
        
            if (i+1)*BATCH<=(len(tokens)//BATCH)*BATCH:
                temp=[]
                temp.append(tokens_sort[BATCH*i:BATCH*(i+1)])
                temp.append(labels_sort[BATCH*i:BATCH*(i+1)])
                temp.append(labels_num_sort[BATCH*i:BATCH*(i+1)])
                temp.append(slots_sort[BATCH*i:BATCH*(i+1)])
            
            else:
                temp=[]
                temp.append(tokens_sort[BATCH*i:BATCH*i+(len(tokens)%BATCH)])
                temp.append(labels_sort[BATCH*i:BATCH*i+(len(tokens)%BATCH)])
                temp.append(labels_num_sort[BATCH*i:BATCH*i+(len(tokens)%BATCH)])
                temp.append(slots_sort[BATCH*i:BATCH*i+(len(tokens)%BATCH)])
            
            data_split.append(temp)

        # get parts of data by BATCH_SIZE

        data_parts=[]
        print("\n### get some information ###")
        for i in range(len(data_split)):
            data_parts.append(self.get_data_list(data_split[i], to_ids))
            # prog.progress_bar(i,len(data_split))

        return data_parts
        
    # tokens to ids
    def bert_ids(self,data_tokens):
        input_id=[self.tokenizer.convert_tokens_to_ids(k) for k in data_tokens]
        return {"input_id":input_id}

    # gain word_emb
    def bert_get_hidden(self,input_id,attention_mask,BATCH=BATCH_SIZE):

        tokens_tensor=torch.tensor([input_id]) 
        tokens_tensor=torch.reshape(tokens_tensor,(BATCH,-1))
        attention_tensor=torch.tensor([attention_mask])
        attention_tensor=torch.reshape(attention_tensor,(BATCH,-1))

        hidden_states=self.model(tokens_tensor,attention_tensor,output_hidden_states=True)
        
        return hidden_states

    # gain word emb by BERT
    def bert_emb(self,ids,attention,BATCH=BATCH_SIZE): 

        emb=self.bert_get_hidden(ids,attention,BATCH)["last_hidden_state"] #embedding by BATCH_SIZE

        # emb=[]
    
        # for i in range (BATCH_SIZE):
        #     emb.append(self.bert_get_hidden(ids[i],attention[i])["last_hidden_state"])
        #     prog.progress_bar(i,BATCH_SIZE,batch_n)

        return emb


