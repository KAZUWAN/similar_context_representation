import numpy as np
import os
from .Data_Making import Dataset_SIM_loading as load
from .Data_Making import Get_parts_of_data as get_parts 
from .config import *
from .Data_Making import sample_progress as prog

# from Data_Making import Dataset_SIM_loading as load
# from Data_Making import Get_parts_of_data as get_parts 
# from config import *
# from Data_Making import check_progress as prog


    # loading data

    # print(os.path.abspath(__file__))
    # print(os.path.dirname(os.path.abspath(__file__)))


    # print(os.path.join(os.path.dirname(os.path.abspath(__file__)),path))
    # print(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path)))

# path example 

def split_data(input_path):

    path=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),input_path))

    data_original=load.json_loading(path)

    tokens=data_original["tokens"]
    labels=data_original["labels"]
    labels_num=data_original["labels_num"]

    # sort data
    length=[len(tokens[i]) for i in range(len(tokens))]
    sort_index=np.argsort(length)
    tokens_sort=[tokens[sort_index[i]] for i in range(len(sort_index))]
    labels_sort=[labels[sort_index[i]] for i in range(len(sort_index))]
    labels_num_sort=[labels_num[sort_index[i]] for i in range(len(sort_index))]


    # split data by BATCH_SIZE
    data_split=[]

    print("##split by BATCH##")
    for i in range((len(tokens)//BATCH_SIZE)+1):
    
        if (i+1)*BATCH_SIZE<=(len(tokens)//BATCH_SIZE)*BATCH_SIZE:
            temp=[]
            temp.append(tokens_sort[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
            temp.append(labels_sort[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
            temp.append(labels_num_sort[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
        
        else:
            temp=[]
            temp.append(tokens_sort[BATCH_SIZE*i:BATCH_SIZE*i+(len(tokens)%BATCH_SIZE)])
            temp.append(labels_sort[BATCH_SIZE*i:BATCH_SIZE*i+(len(tokens)%BATCH_SIZE)])
            temp.append(labels_num_sort[BATCH_SIZE*i:BATCH_SIZE*i+(len(tokens)%BATCH_SIZE)])
        
        data_split.append(temp)

        return data_split

    # get parts of data by BATCH_SIZE

    # data_parts=[get_parts.data_list(data_split[i]) for i in range(len(data_split))] 

    # data_parts=[]
    # print("##get some information##")
    # for i in range(len(data_split)):
    #     data_parts.append(get_parts.data_list(data_split[i]))
    #     prog.progress_bar(i,len(data_split))


    # return data_parts

# data_parts=split_data("/home/g1923037/laboratory/share/Data/simulated-dialogue-master/sim-M/train.json")
# print("len data_parts:",len(data_parts))
# print("keys data_parts[0]",data_parts[0].keys())
# print("len data_parts[0][tokens]:",len(data_parts[0]["tokens"]))
# print("len data_parts[-1][tokens]",len(data_parts[-1]["tokens"]))


