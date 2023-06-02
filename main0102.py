import os
import time
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import tracemalloc
from memory_profiler import profile
import datetime
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.config import *
from src.Data_Making import check_progress as prog
from src.semantic_sentence_embed01 import MakeSematicSentenceEmbedding
from src.util import load_data, make_datacollection02, make_data_forBatch03, add_special_token, calculate_cos_sim


# @profile
def train():

    # tracemalloc.start()

    # load dataset
    path_train_data = DATA_TRAIN_PATH
    input_path_tr = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_train_data))
    train_dataset = load_data.load_json_data(input_path_tr)

    # make data collection
    creater = make_datacollection02.CreateOriginal(train_dataset)
    creater.remove_only_o()
    creater.create_sentence_list()
    creater.create_sentences_dict()
    all_sentences = creater.return_something()
    print("length all sentences :",len(all_sentences["sentence_list"]))

    # initialize model
    generate_semantic_sentence_embedding_model = MakeSematicSentenceEmbedding()
    # initialize optimizer
    optimizer = optim.Adam(generate_semantic_sentence_embedding_model.parameters(), lr=Learning_Rate)

    for epoch in range(N_EPOCH):
        # make BATCH data
        # for one epoch
        batch_creater = make_data_forBatch03.BatchManager(all_sentences["sentence_list"], all_sentences["slot_list"],\
            all_sentences["sentence_dict"], creater.sentence_only_o)

        batch_creater.create_anchor_positive_negative()
        batch_data = copy.copy(batch_creater.return_batch_data())

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
                data_train.append(add_special_token.add_specialtokens(batch_data[j], max_length,plus_token=True, padding=True))    
        
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
        # print("anchor_normalized\n", anchor_normalized[0:3])
        # print("positive_normalized\n", positive_normalized[0:3])
        # print("negative_normalized\n", negative_normalized[0:3])
        print(f"[EPOCH{epoch}] loss: {triplet_loss.item()}")
        # print("norm anchor: ", torch.linalg.norm(embedding_pred_list[0], dim=1))
        # print("normalized anchor : ", anchor_normalized)

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics("lineno")

        # print("top 5")
        # for stat in top_stats[:5]:
        #     print(stat)

    print("Shape Semantic Sentence Embeddings :",embedding_pred_list[0].shape)

    return generate_semantic_sentence_embedding_model

def eval(generate_semantic_setntence_embedding_model):

    log_file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./log/log_evaluation_cos_similarity.txt"))
    log = open(log_file_path, "a")

    current_time = datetime.datetime.today()

    log.write(f"\n\n{current_time}")

    log.write("\n###  load data  ###")
     # load dataset
    path_train_data = DATA_TRAIN_PATH
    path_test_data = DATA_TEST_PATH
    input_path_tr=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_train_data))
    input_path_te=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),path_test_data))
    train_dataset = load_data.load_json_data(input_path_tr)
    test_dataset = load_data.load_json_data(input_path_te)

    # make data collection
    creater_train = make_datacollection02.CreateOriginal(train_dataset)
    creater_train.remove_only_o()
    creater_train.create_sentence_list()
    creater_train.create_sentences_dict()
    train_sentences = creater_train.return_something()
    temp_length_sentences = len(train_sentences["sentence_list"])
    log.write(f"\n##  length train sentences : {temp_length_sentences}")
    
    creater_test = make_datacollection02.CreateOriginal(test_dataset)
    creater_test.remove_only_o()
    creater_test.create_sentence_list()
    creater_test.create_sentences_dict()
    test_sentences = creater_test.return_something()
    temp_length_sentences = len(test_sentences["sentence_list"])
    log.write(f"\n##  length test sentences : {temp_length_sentences}")

    #  initialize 未学習オリジナルモデル
    original_model = MakeSematicSentenceEmbedding()
    
    #  学習済みモデル
    trained_model = generate_semantic_setntence_embedding_model

    # オリジナルBERTからのpoolingによる分散表現を保存
    original_emb_dataset = []
    log.write(f"\nlen train_sentences:{len(train_sentences['sentence_list'])}")
    for i in range(len(train_sentences["sentence_list"])):
        temp_tokens_and_mask = add_special_token.add_specialtokens([train_sentences["sentence_list"][i]], len(train_sentences["sentence_list"][i]),plus_token=False, padding=False)
        original_emb_dataset.append(F.normalize(original_model(temp_tokens_and_mask["specialtokens_added"], temp_tokens_and_mask["attention_mask_list"])))        

    # ファインチューニングしたBERTからのpoolingによる分散表現の保存
    trained_emb_dataset = []
    for i in range(len(train_sentences["sentence_list"])):
        temp_tokens_and_mask = add_special_token.add_specialtokens([train_sentences["sentence_list"][i]], len(train_sentences["sentence_list"][i]),plus_token=False, padding=False)
        trained_emb_dataset.append(F.normalize(trained_model(temp_tokens_and_mask["specialtokens_added"], temp_tokens_and_mask["attention_mask_list"]))) 

    # オリジナルBERTからのpoolingによる評価データの分散表現の獲得，類似度比較
    # とりあえず１文に対して

    log.write(f"\n\n###  evaluation  ###")

    top = 5
    eval_num = 10
    log.write("\n\n###  original model  ###")
    top_sim = calculate_cos_sim.CaluculateCosSim()
    for i in range(eval_num):
        temp_tokens_and_mask = add_special_token.add_specialtokens([test_sentences["sentence_list"][i]], len(test_sentences["sentence_list"][i]),plus_token=False, padding=False)
        test_original_emb = F.normalize(original_model(temp_tokens_and_mask["specialtokens_added"], temp_tokens_and_mask["attention_mask_list"]))
        top_sim.top_cos_sim(test_original_emb, original_emb_dataset,top) #calculate similarity
        temp_sentence = test_sentences["sentence_list"][i]
        temp_slot = test_sentences["slot_list"][i]
        log.write(f"\n\ninput sentence: {temp_sentence}, slot: {temp_slot}")
        log.write(f"\nslot: {creater_test.ori_slot_list[i]}")
        log.write(f"\nsimilarity top{top}")
        for j in range(top):
            temp_sentence = train_sentences["sentence_list"][top_sim.top_sentence_index[j]]
            temp_slot = train_sentences["slot_list"][top_sim.top_sentence_index[j]]
            log.write(f"\ntop{j+1}: {temp_sentence}, :slot: {temp_slot}, similarity: {top_sim.top_cos[j]}")
            # log.write(f"{\ncreater_train.ori_sentence_list[top_sim.top_sentence_index[j]]}")
            log.write(f"\n{creater_train.ori_slot_list[top_sim.top_sentence_index[j]]}")
    # ファインチューニングしたpoolingによる評価データの分散表現の獲得，類似度比較
    # とりあえず１文に対して
    log.write("\n\n###  trained model  ###")
    top_sim = calculate_cos_sim.CaluculateCosSim()
    for i in range(eval_num):
        temp_tokens_and_mask = add_special_token.add_specialtokens([test_sentences["sentence_list"][i]], len(test_sentences["sentence_list"][i]),plus_token=False, padding=False)
        test_trained_emb = F.normalize(trained_model(temp_tokens_and_mask["specialtokens_added"], temp_tokens_and_mask["attention_mask_list"]))
        top_sim.top_cos_sim(test_trained_emb, trained_emb_dataset, top) #calculate similarity
        temp_sentence = test_sentences["sentence_list"][i]
        temp_slot = test_sentences["slot_list"][i]
        log.write(f"\n\ninput sentence: {temp_sentence}, slot: {temp_slot}")
        log.write(f"\nslot: {creater_test.ori_slot_list[i]}")
        log.write(f"\nsimilarity top{top}")
        for j in range(top):
            temp_sentence = train_sentences["sentence_list"][top_sim.top_sentence_index[j]]
            temp_slot = train_sentences["slot_list"][top_sim.top_sentence_index[j]]
            log.write(f"\ntop{j+1}: {temp_sentence}, :slot: {temp_slot}, similarity: {top_sim.top_cos[j]}")
            # log.write(f"{\ncreater_train.ori_sentence_list[top_sim.top_sentence_index[j]]}")
            log.write(f"\n{creater_train.ori_slot_list[top_sim.top_sentence_index[j]]}")
    
    log.close()

    #  ファインチューニング前のオリジナルも比較すべき

if __name__ == "__main__":
    time_start = time.perf_counter()
    torch.manual_seed(0)
    # trained_model = train()
    # torch.save(trained_model.state_dict(),f'src/trained_model/{DATASETS}_generate_semantic_sentence_embedding_model0102.pth')
    trained_model = MakeSematicSentenceEmbedding()
    trained_model.load_state_dict(torch.load(f'src/trained_model/{DATASETS}_generate_semantic_sentence_embedding_model0102.pth'))
    eval(trained_model)
    time_end = time.perf_counter()
    print(f"##  process time(minutes): {(time_end-time_start)/60.0}")