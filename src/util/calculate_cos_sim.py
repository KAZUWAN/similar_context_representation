import numpy as np
import torch.nn.functional as F
import os
from src.config import *

class CaluculateCosSim():
    def __init__(self) -> None:
        self.top_cos = None
        self.top_sentence_index = None
        return

    def top_cos_sim(self, target_sentence_embedding, dataset_sentence_embedding_list, top=3):
        self.top_cos = np.zeros(top+1)
        self.top_sentence_index = np.zeros(top+1, dtype=int)
        for i in range(len(dataset_sentence_embedding_list)):
            # print(f"shape embedding:{target_sentence_embedding.shape}")
            # print(f"shape embedding:{dataset_sentence_embedding_list[i].shape}")
            temp_cos_sim = F.cosine_similarity(target_sentence_embedding, dataset_sentence_embedding_list[i], dim=1)
            # print(f"tmep_cos_sim: {temp_cos_sim.shape}")
            self.top_cos[top] = temp_cos_sim
            self.top_sentence_index[top] = i
            temp_sort_index = np.argsort(-self.top_cos) #類似度降順
            self.top_cos = self.top_cos[temp_sort_index]
            self.top_sentence_index = self.top_sentence_index[temp_sort_index]
        path_log = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),logprint_path))
        with open(path_log, mode = "a") as logf:
            print("\ntop_cos: ", self.top_cos[:top], file= logf)
            print(f"top_cos_sentence_index: {self.top_sentence_index[:top]}", file= logf)
        return





