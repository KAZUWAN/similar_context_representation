import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import collections
import torch
import torch.nn.functional as F
import os
from src.util import add_special_token
from src.Data_Making import check_progress as prog
from src.config import *

class KNNManager():
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy_list = []
        self.precision_list = []
        self.recall_list = []

    def KNN(self, n_neighbors=3):
        knn = KNeighborsClassifier(n_neighbors = n_neighbors)
        knn.fit(self.x_train, self.y_train)
        y_pred = knn.predict(self.x_test)
        confusion = confusion_matrix(self.y_test, y_pred)
        path_log = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),logprint_path))
        with open(path_log, mode = "a") as logf:
            print(f'\nconfusion k={n_neighbors}:', file= logf)
            print("# prediction\nt\nr\nu\ne", file= logf)
            print(confusion, file= logf)
        self.accuracy_list.append(accuracy_score(self.y_test, y_pred))
        self.precision_list.append(precision_score(self.y_test, y_pred, average="macro"))
        self.recall_list.append(recall_score(self.y_test, y_pred, average="macro"))

    def run_knn(self, title, k_range_start=1, k_range_end=100, k_range_step=1):
        self.x_train = np.array([i.detach().numpy()  for i in self.x_train])
        self.x_test = np.array([i.detach().numpy()  for i in self.x_test])
        k_range = range(k_range_start, k_range_end, k_range_step)
        for k in k_range:
            self.KNN(n_neighbors = k)
        plt.clf()
        plt.title(title, fontsize=16)
        plt.plot(k_range, self.accuracy_list, label="accuracy")
        plt.plot(k_range, self.precision_list, label="precision")
        plt.plot(k_range, self.recall_list, label="recall")
        plt.xlabel("k", fontsize=16)
        plt.xticks(k_range, fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="best", fontsize=16)
        now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig(f"./log/{title}_knn_{k_range_start}_{k_range_end}_{k_range_step}_result_{now_str}.png")
        path_log = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),logprint_path))
        with open(path_log, mode = "a") as logf:
            print(f"len(accuracy list){len(self.accuracy_list)}", file= logf)
            plt.show()
            max_accuracy = max(self.accuracy_list)
            index        = self.accuracy_list.index(max_accuracy)
            best_k_range = k_range[index]
            print("k="+str(best_k_range)+": accuracy max; "+str(max_accuracy), file= logf)

            max_precision = max(self.precision_list)
            index        = self.precision_list.index(max_precision)
            best_k_range = k_range[index]
            print("k="+str(best_k_range)+": precision max; "+str(max_precision), file= logf)

            max_recall = max(self.recall_list)
            index        = self.recall_list.index(max_recall)
            best_k_range = k_range[index]
            print("k="+str(best_k_range)+": recall max; "+str(max_recall), file= logf)

        # def data_arrangement(self, train_emb_list, train_slot_list, test_emb_list, test_slot_list):
        #     self.x_train = []
        #     self.y_train = []
        #     self.x_test = []
        #     self.y_test = []
        #     print(f"length train emb list: {len(train_emb_list)}")
        #     print(f"length test emb list: {len(test_emb_list)}")
        #     print(f"length train slot list: {len(train_slot_list)}")
        #     print(f"length test slot list: {len(test_slot_list)}")
            
        #     # data_train_slot = sum(train_slot_list, [])
        #     # data_test_slot = sum(test_slot_list, [])
        #     # count_train = collections.Counter(data_train_slot)
        #     # count_test = collections.Counter(data_test_slot)
        #     # print(f"count train : {count_train}")
        #     # print(f"count test : {count_test}")

        #     print("train_emb_list[10].shape")
        #     print(train_emb_list[10].shape)
        #     print(train_emb_list[10][0].shape)
        #     print("train_emb_list[0].shape")
        #     print(train_emb_list[0].shape)
        #     print("train_emb_list[0][1].shape")
        #     print(train_emb_list[0][1].shape)
        #     for i in range(len(train_emb_list)):
        #         for j in range(len(train_slot_list[i])):
        #             if train_slot_list[i][j] != 'o':
        #                 print(f"i={i}, j={j}")
        #                 self.x_train.append(train_emb_list[i][j])
        #                 self.y_train.append(train_slot_list[i][j])
        #     for i in range(len(test_slot_list)):
        #         for j in range(len(test_emb_list[i])):
        #             if test_slot_list[i][j] != "o":
        #                 self.x_test.append(test_emb_list[i][j])
        #                 self.y_test.append(test_slot_list[i][j])

        #     count_y_train = collections.Counter(self.y_train)
        #     count_y_test = collections.Counter(self.y_test)
        #     print(f"count y train :{count_y_train}")
        #     print(f"count y test :{count_y_test}")

        #     print(f"訓練データの単語数：{len(self.y_train)}")
        #     print(f"テストデータの単語数：{len(self.y_test)}")



    def data_arrangement(self, model, train_sentences, test_sentences, train_slot_list, test_slot_list):
            self.x_train = []
            self.y_train = []
            self.x_test = []
            self.y_test = []
            path_log = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),logprint_path))
            with open(path_log, mode = "a") as logf:
                print(f"length train slot list: {len(train_slot_list)}", file= logf)
                print(f"length test slot list: {len(test_slot_list)}", file= logf)
            
                data_train_slot = sum(train_slot_list, [])
                data_test_slot = sum(test_slot_list, [])
                count_train = collections.Counter(data_train_slot)
                count_test = collections.Counter(data_test_slot)
                print(f"count train : {count_train}", file= logf)
                print(f"count test : {count_test}\n", file= logf)

            # ファインチューニングしたBERTからのpoolingによる分散表現の保存 train
            for i in range(len(train_sentences)):
                temp_tokens_and_mask = add_special_token.add_specialtokens([train_sentences[i]], len(train_sentences[i]), plus_token=False, padding=False)
                temp_emb = F.normalize(model(temp_tokens_and_mask["specialtokens_added"], temp_tokens_and_mask["attention_mask_list"], word_emb=True)[1])
                # print(f"temp_emb : {temp_emb.shape}")
                temp_emb = torch.reshape(temp_emb, (-1, 768))
                # print(f"temp_emb : {temp_emb.shape}")
                # print(f"train_slot:{len(train_slot_list[i])}")
                for j in range(len(train_slot_list[i])):
                    if train_slot_list[i][j] != 'o': # 文脈後以外を訓練データに取り込む
                        # print(f"i,j: {i},{j}")
                        self.x_train.append(temp_emb[j])
                        self.y_train.append(train_slot_list[i][j])
                    # prog.progress_bar(int(i//(30)), int(len(train_sentences))//(30))

            # ファインチューニングしたBERTからのpoolingによる分散表現の保存 test  
            for i in range(len(test_sentences)):
                temp_tokens_and_mask = add_special_token.add_specialtokens([test_sentences[i]], len(test_sentences[i]), plus_token=False, padding=False)
                temp_emb = F.normalize(model(temp_tokens_and_mask["specialtokens_added"], temp_tokens_and_mask["attention_mask_list"], word_emb=True)[1])
                temp_emb = torch.reshape(temp_emb, (-1, 768))
                for j in range(len(test_slot_list[i])):
                    if test_slot_list[i][j] != 'o':
                        self.x_test.append(temp_emb[j])
                        self.y_test.append(test_slot_list[i][j])
                    # prog.progress_bar(int(i//(30)), int(len(test_sentences))//(30))

            count_y_train = collections.Counter(self.y_train)
            count_y_test = collections.Counter(self.y_test)
            with open(path_log, mode = "a") as logf:
                print(f"\ncount y train :{count_y_train}", file= logf)
                print(f"count y test :{count_y_test}", file= logf)

                print(f"train data words num:{len(self.y_train)}", file= logf)
                print(f"test data words num{len(self.y_test)}", file= logf)
