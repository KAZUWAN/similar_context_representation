import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.value_detector import ValueDetector
from src.config import *
from src.Data_Making.Bert_func import BertFunction
from src.Data_Making.check_progress import progress_bar as prog



def train():

    # initialize model
    value_detector = ValueDetector()
    # initialize optimizer
    optimizer = optim.AdamW(value_detector.parameters())

    # load dataset
    xs = torch.load("Data/emb_train_sim_M.pt")
    ys = torch.tensor(np.load("Data/one_hot_train.npy"))
    ys = torch.argmax(ys, dim = -1)
    masks = torch.tensor(np.load("Data/attention_mask_array.npy"))
    N = xs.shape[0]

    # edit kazuya
    path_train_data="./Data/simulated-dialogue-master/sim-M/train.json"
    path_test_data="./Data/simulated-dialogue-master/sim-M/test.json"
    input_path_tr=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_train_data))
    input_path_te=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_test_data))

    Functions=BertFunction()

    data_train=Functions.split_data(input_path_tr)
    data_test=Functions.split_data(input_path_te)
    
    print("\n### get word_emb by BERT ###")

    x_train=[Functions.bert_emb(data_train[i]["input_id"],data_train[i]["attention_mask"],i) for i in range(len(data_train)-1)]
    x_test=[Functions.bert_emb(data_test[i]["input_id"],data_test[i]["attention_mask"],i) for i in range(len(data_test)-1)]

    y_train=[data_train[i]["labels_num_add"] for i in range(len(data_train))]
    y_test=[data_test[i]["labels_num_add"] for i in range(len(data_test))]

    assert False



    for epoch in range(N_EPOCH):
        for n in range(0,(N//BATCH_SIZE)*BATCH_SIZE,BATCH_SIZE):
            x = xs[n:n+BATCH_SIZE] # (BATCH_SIZE, T, D_BERT)
            y = ys[n:n+BATCH_SIZE] # (BATCH_SIZE, T, N_CLASS=3)
            mask = masks[n:n+BATCH_SIZE] # (BATCH_SIZE, T)
            # predict labels
            y_pred = value_detector(x, mask = mask) # (BATCH_SIZE, T, N_CLASS=3)
            # compute loss
            y_pred = y_pred.reshape(BATCH_SIZE * T, N_CLASS)
            y = y.reshape(BATCH_SIZE * T,)
            loss = F.cross_entropy(y_pred, y)
            # reset gradient
            optimizer.zero_grad()
            # compute gradient
            loss.backward()
            # update parameter
            optimizer.step()
        print(f"[EPOCH{epoch}] loss: {loss.item()}")


if __name__ == "__main__":
    torch.manual_seed(0)
    train()
