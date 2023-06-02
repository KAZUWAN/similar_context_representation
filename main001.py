import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.value_detector import ValueDetector
from src.config import *


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
