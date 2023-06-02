import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *

class ContextEstimator(nn.Module):
    def __init__(self,D_feature): #D_feature = D_Glove or D-word2
        super().__init__()
        self.fc1 = nn.Linear(D_feature, D_feature, bias = True)
        self.fc2 = nn.Linear(D_feature, D_feature, bias = True)
        self.fc3 = nn.Linear(D_feature, 1, bias = True)

    def forward(self, x):
        h1 = self.fc1(x)
        hc = self.fc2(h1)
        h3 = self.fc3(hc)
        nn.Sigmoid(dim=-1)(h3)
        # 確率が0.5より大きいかどうかでラベルを与える必要がある

        return hc #おそらく文脈語かどうかの判定をスロット推定器の学習時に使用する必要がある