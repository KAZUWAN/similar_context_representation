import os
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.config import *
from src.Data_Making import check_progress as prog
from src.semantic_sentence_embed01 import MakeSematicSentenceEmbedding
from src.util import load_data, make_datacollection, make_data_forBatch, add_special_token
from transformers import BertModel


def train():

    # load dataset
    path_train_data="./Data/simulated-dialogue-master/sim-M/train.json"
    # path_test_data="./Data/simulated-dialogue-master/sim-M/test.json"
    input_path_tr=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_train_data))
    # input_path_te=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_test_data))
    train_dataset = load_data.load_json_data(input_path_tr)

    # make data collection
    creater = make_datacollection.CreateOriginal(train_dataset)
    creater.remove_only_o()
    creater.create_sentence_list()
    creater.create_sentences_dict()
    all_sentences = creater.return_something()
    print("length all sentences :",len(all_sentences["sentence_list"]))

    # initialize model
    generate_semantic_sentence_embedding_model = MakeSematicSentenceEmbedding()
    # initialize optimizer
    optimizer = optim.Adam(generate_semantic_sentence_embedding_model.parameters())

    # temp_merge_sentence_slot = copy.copy(all_sentences["sentence_list"])
    # for i in range(len(all_sentences["sentence_list"])):
    #     temp_merge_sentence_slot[i].insert(0,all_sentences["slot_list"][i])
    # print("temp_merge_sentence_slot", temp_merge_sentence_slot[0:4])
    # この後，バッチを作成した後のセンテンスの最初を見て適切に作成されているか確認
    # all_sentences["sentence_list"]の代わりにtemp_merge_sentence_slotを渡す


    for epoch in range(N_EPOCH):
        # make BATCH data
        # for one epoch
        batch_creater = make_data_forBatch.BatchManager(all_sentences["sentence_list"], all_sentences["slot_list"],\
            all_sentences["sentence_dict"])

        batch_creater.create_anchor_positive_negative()
        batch_data = copy.copy(batch_creater.return_batch_data())
        # print("anchor_sentence_list\n", batch_data["anchor_sentence_list"][0:3])
        # print("positive_sentence_list\n", batch_data["positive_sentence_list"][0:3])
        # print("negative_sentence_list\n", batch_data["negative_sentence_list"][0:3])

        temp_length_list = []
        for i in batch_data.keys():
            if i != "anchor_slot_list":
                temp_length_list.extend([len(j) for j in batch_data[i]])
        # print(len(temp_length_list))
        # print("length list:",temp_length_list)
        max_length = max(temp_length_list)
        # print("max Length:",max_length)

        data_train = []
        for j in batch_data.keys():
            if j != "anchor_slot_list":
                data_train.append(add_special_token.add_specialtokens(batch_data[j], max_length, padding=True))    
        
        # data_train_anchor = data_train[0]
        # data_train_positive = data_train[1]
        # data_train_negative = data_train[2]
        # temp_keys = ["anchor_embedding_list", "positive_embedding_list", "negative_embedding_list"]
        # print("len data_train[0]:",len(data_train[0]["specialtokens_added"]))
        embedding_pred_list = [generate_semantic_sentence_embedding_model(data_train[k]["specialtokens_added"], data_train[k]["attention_mask_list"]) for k in range(3)]
        
        # normalization
        
        anchor_normalized = F.normalize(embedding_pred_list[0], dim=1)
        positive_normalized = F.normalize(embedding_pred_list[1], dim=1)
        negative_normalized = F.normalize(embedding_pred_list[2], dim=1)

        triplet_loss = F.triplet_margin_loss(anchor_normalized, positive_normalized, negative_normalized, margin = 1.0)
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
        print("anchor_normalized\n", anchor_normalized[0:3])
        print("positive_normalized\n", positive_normalized[0:3])
        print("negative_normalized\n", negative_normalized[0:3])
        print(f"\n[EPOCH{epoch}] loss: {triplet_loss.item()}\n")
        # print("norm anchor: ", torch.linalg.norm(embedding_pred_list[0], dim=1))
        # print("normalized anchor : ", anchor_normalized)
    print("Shape Semantic Sentence Embeddings :",embedding_pred_list[0].shape)

    return generate_semantic_sentence_embedding_model

    # def eval(generate_semantic_setntence_embedding_model):
    #     print("\n###  load data test  ###")
    #     # # load dataset
    #     # path_train_data="./Data/simulated-dialogue-master/sim-M/train.json"
    #     # # path_test_data="./Data/simulated-dialogue-master/sim-M/test.json"
    #     # input_path_tr=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_train_data))
    #     # # input_path_te=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_test_data))
    #     # train_dataset = load_data.load_json_data(input_path_tr)
    #     # # make data collection
    #     # creater = make_datacollection.CreateOriginal(train_dataset)
    #     # creater.remove_only_o()
    #     # creater.create_sentence_list()
    #     # creater.create_sentences_dict()
    #     # all_sentences = creater.return_something()
    #     # print("length all sentences :",len(all_sentences["sentence_list"]))

    #     #  オリジナルのBERTモデルの潜在表現での訓練データの分散表現のデータ集を作成
    #     Bert_model = BertModel.from_pretrained(pretrained_name)
    #     Bert_emb = 

    #     # 作成したモデルの潜在表現での訓練データの分散表現のデータ集を作成
    #     Semntic_sentence_embedding = generate_semantic_setntence_embedding_model(ids, attention_mask)

    #     # さらに，それぞれに対して評価データの分散表現を獲得
    #     # それとデータ集との類似度が最も近いものを取り出し，その違いを比較



        


if __name__ == "__main__":
    torch.manual_seed(0)
    model = train()
    # torch.save(model.state_dict(),'src/trained_model/value_detector.pth')
    # trained_model = MakeSematicSentenceEmbedding()
    # trained_model.load_state_dict(torch.load('src/trained_model/value_detector.pth'))
    # eval(trained_model)
    # load_slots()
