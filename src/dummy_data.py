import torch
from .config import *

def get_dummy_data():
    x = torch.randn((BATCH_SIZE, T, D_BERT))
    y = torch.randint(N_CLASS, size = (BATCH_SIZE * T,))
    y = torch.eye(N_CLASS)[y]
    y = y.reshape((BATCH_SIZE, T, N_CLASS))
    return x, y
