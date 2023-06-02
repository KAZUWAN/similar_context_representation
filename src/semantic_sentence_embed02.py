
# make like sentenceBERT

# make k-means for clustering of semantic sentence embeddings
# or kNN

from sentence_transformers import models
from transformers import BertTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
from .config import *
from .Data_Making import check_progress as prog


class MakeSematicSentenceEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_name)
        self.model = AutoModel.from_pretrained(pretrained_name)
        # self.config = AutoConfig.from_pretrained(pretrained_name)
        self.pooling_model = models.Pooling(D_BERT)
        
        # def forward(self, features: Dict[str, Tensor]):
        #     token_embeddings = features['token_embeddings']
        #     attention_mask = features['attention_mask'] #poolingのforwardの型，辞書型になっている．直前でいいので，キーを与えてあげるのがよさそう

        
  # tokens to ids
    def bert_ids(self,sentence_list):
        # print("sentence list 0to2",sentence_list[0:3])
        input_ids_list=[self.tokenizer.convert_tokens_to_ids(k) for k in sentence_list]
        
        return {"input_ids_list":input_ids_list}  
    
    def bert_emb(self, input_ids_list, attention_mask_list): 
        emb=self.bert_get_hidden(input_ids_list, attention_mask_list)["last_hidden_state"] #embedding by BATCH_SIZE

        return emb

    def bert_get_hidden(self,input_ids_list,attention_mask_list):
        tokens_tensor=torch.tensor(input_ids_list) 
        # print(f"shape input ids list : {len(input_ids_list)}")
        tokens_tensor=torch.reshape(tokens_tensor,(len(input_ids_list),-1))
        attention_tensor=torch.tensor(attention_mask_list)
        attention_tensor=torch.reshape(attention_tensor,(len(input_ids_list),-1))
        hidden_states=self.model(tokens_tensor,attention_tensor,output_hidden_states=True)
        
        return hidden_states

    def forward(self, sentence_list, attention_mask_list, word_emb=False):
        input_ids_list = self.bert_ids(sentence_list)["input_ids_list"]
        # print("len input_ids_list", len(input_ids_list))
        # print(input_ids_list[0:3])
        sentence_embedding = self.bert_emb(input_ids_list, attention_mask_list)
        # print("len sentence_embedding_list", len(sentence_embedding))
        # print("sentence_embedding tensor shape:",sentence_embedding.shape)
        attention_mask_tensor = torch.tensor(attention_mask_list)
        # print("attention mask tensor shape:", attention_mask_tensor.shape)
        feature = {"token_embeddings":sentence_embedding, "attention_mask":attention_mask_tensor }
        semantic_dict = self.pooling_model(feature)
        # print("type semantic : ",type(semantic_dict))
        # print("semantic keys : ",semantic_dict.keys())
        # print("len semantic",len(semantic_dict))
        semantic_sentence_embedding = semantic_dict["sentence_embedding"]

        if word_emb == True:
            return (semantic_sentence_embedding, sentence_embedding)
        return semantic_sentence_embedding


# 下記サイト参考　：　BERT部分のパラメータ更新のためのファインチューニング部分
# https://www.ai-shift.co.jp/techblog/2145




