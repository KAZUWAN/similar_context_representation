
from . import Add_tokens
from . import One_hot
from . import Convert_tokens





# get data to use laerning
# inputs (data of sentences)
# shape = (BATCH_SIZE,3) # 3 -> (tokens,labels,labels_num)

def data_list(data):
    tokens=data[0]
    labels=data[1]
    labels_num=data[2]

    # add tokens for BERT tokenizer
    data_add=Add_tokens.add_tokens(tokens,labels,labels_num)
    tokens_add=data_add["tokens_add"]
    labels_add=data_add["labels_add"]
    labels_num_add=data_add["labels_num_add"]
    attention_mask=data_add["attention_mask"]

    # convert class label to one-hot
    one_hot_labels=One_hot.one_hot(labels_num_add)
    one_hot_labels_arr=one_hot_labels["one_hot_"]

    # convert tokens wiht Bert_tokenizer  
    bert_input_id=Convert_tokens.bert_ids(tokens_add)
    input_id=bert_input_id["input_id"]

    return({"data":data,"tokens":tokens,"labels":labels,"label_n":labels_num,"tokens_add":tokens_add,"labels_add":labels_add\
        ,"attention_mask":attention_mask,"labels_num_add":labels_num_add,"one_hot_":one_hot_labels_arr,"input_id":input_id})