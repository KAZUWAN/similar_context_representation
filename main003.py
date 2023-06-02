import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.value_detector02 import ValueDetector
from src.config import *
from src.Data_Making.Bert_func import BertFunction
from src.Data_Making import check_progress as prog



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

    print("\n### load data train ###")
    x_train=[]
    y_train=[]
    masks_train=[]
    label0_n=0
    label1_n=0
    label2_n=0
    label3_n=0
    for i in range(31):  # 31 <- data train number
        path_x_train=f"./Data/Sim_M/x_train/x_train_{i}.pt"
        input_path_x_tr=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_x_train))
        path_y_train=f"./Data/Sim_M/y_train/y_train_{i}.npy"
        input_path_y_tr=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_y_train))
        path_masks_train=f"./Data/Sim_M/masks_train/masks_train_{i}.npy"
        input_path_masks_tr=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_masks_train))

        temp_y=np.load(input_path_y_tr)
        label0_n+=np.sum(np.where(temp_y==0, 1, 0))
        label1_n+=np.sum(np.where(temp_y==1, 1, 0))
        label2_n+=np.sum(np.where(temp_y==2, 1, 0))
        label3_n+=np.sum(np.where(temp_y==3, 1, 0))
        x_train_temp=torch.load(input_path_x_tr)
        x_train.append(x_train_temp)
        y_train_temp=torch.tensor(np.load(input_path_y_tr))
        y_train.append(y_train_temp)
        masks_train_temp=torch.tensor(np.load(input_path_masks_tr))
        masks_train.append(masks_train_temp)
    print("label 0: ",label0_n)
    print("label 1: ",label1_n)
    print("label 2: ",label2_n)
    print("label 3: ",label3_n)

    weight_loss=torch.tensor([(label0_n+label1_n+label2_n)/label0_n, (label0_n+label1_n+label2_n)/label1_n, (label0_n+label1_n+label2_n)/label2_n, (label0_n+label1_n+label2_n)/label3_n])
    print("loss weights : ", weight_loss)
    print(weight_loss.shape)

    print("\n### load data test ###")
    x_test=[]
    y_test=[]
    masks_test=[]
    for i in range(22):  # 22 <- data test number
        path_x_test=f"./Data/Sim_M/x_test/x_test_{i}.pt"
        input_path_x_te=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_x_test))
        path_y_test=f"./Data/Sim_M/y_test/y_test_{i}.npy"
        input_path_y_te=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_y_test))
        path_masks_test=f"./Data/Sim_M/masks_test/masks_test_{i}.npy"
        input_path_masks_te=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_masks_test))

        x_test_temp=torch.load(input_path_x_te)
        x_test.append(x_test_temp)
        y_test_temp=torch.tensor(np.load(input_path_y_te))
        y_test.append(y_test_temp)
        masks_test_temp=torch.tensor(np.load(input_path_masks_te))
        masks_test.append(masks_test_temp)

    print("\n### training... ###")
    for epoch in range(N_EPOCH):
        for n in range(len(x_train)-1):
            x = x_train[n] # (BATCH_SIZE, Time_step, D_BERT)
            y = y_train[n] # (BATCH_SIZE, Time_step)
            mask = masks_train[n] # (BATCH_SIZE, Time_step)
            seq_len=len(x_train[n][0])
            
            # predict labels
            y_pred = value_detector(x, mask = mask) # (BATCH_SIZE, Time_step, N_CLASS=4)
            print(y_pred[0])
            print(y[0])
            # compute loss
            y_pred = y_pred.reshape(BATCH_SIZE * seq_len, N_CLASS)
            y = y.reshape(BATCH_SIZE * seq_len)
            loss = F.cross_entropy(y_pred, y, weight=weight_loss.float(), ignore_index = 3) # len(weight)=N_CLASS
            # reset gradient
            optimizer.zero_grad()
            # compute gradient
            loss.backward()
            # update parameter
            optimizer.step()
            
            # prog.progress_bar(n, len(x_train)-1, epoch)
        print(f"[EPOCH{epoch}] loss: {loss.item()}")


if __name__ == "__main__":
    torch.manual_seed(0)
    train()
