import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import *
from .value_detector02 import Encoder


class SlotEstimator(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder1 =Encoder()
        self.encoder2 =Encoder()
        self.encoder3 =Encoder()
        self.fc1 = nn.Linear(feature_dim, fearture_dim*2, bias = True)
        self.fc2 = nn.Linear(feature_dim, N_SLOT, bias = True)

    def embed(self, p_k, e_bert):
        return 




        