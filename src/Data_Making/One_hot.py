import numpy as np
from keras.utils import np_utils

def one_hot(data_labels_num):
    one_hot_labels=[]

    for i in range(len(data_labels_num)):
        one_hot_labels.append(np_utils.to_categorical(data_labels_num[i],4))

    one_hot_labels=np.array(one_hot_labels)
    one_hot_labels_arr=np.copy(one_hot_labels)
    
    return {"one_hot_":one_hot_labels_arr}