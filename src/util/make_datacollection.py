from copy import copy
import collections

class CreateOriginal():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.ori_sentence_list = None
        self.ori_label_num_list = None
        self.ori_slot_list = None
        self.sentence_list = None
        self.slot_list = None
        self.slot_type = None
        self.sentence_dict = None
        
        return

    def remove_only_o(self): # 文脈語のみの文の削除 #ついでにスロットの種類とかも獲得する
        self.ori_sentence_list = self.dataset["tokens"]
        # labels = dataset["labels"]
        self.ori_label_num_list = self.dataset["labels_num"]
        self.ori_slot_list = self.dataset["slots"]

        count_i = collections.Counter(sum (self.ori_slot_list,[])) #slotsリストを１次元にし，そのスロット集で各スロットの出現頻度を計測 #文脈のみの文の削除前
        r = len(self.ori_label_num_list) #センテンス数
        for i in range(r): #データ数の分だけ確認する
            if self.ori_label_num_list[r-1-i].count(1) == 0: #ラベル番号1  無いとき取り除く．スロットを持つ場合必ずラベルBをもつから #後ろの番号から調べる
                self.ori_sentence_list.pop(r-1-i)
                # labels.pop(r-1-i)
                self.ori_label_num_list.pop(r-1-i)
                self.ori_slot_list.pop(r-1-i)
        # self.slot_list = []
        # for j in self.ori_slot_list:
        #     if j != "o":
        #         self.slot_list.append(j)
        
        count_e = collections.Counter(sum(self.ori_slot_list,[])) #slotsリストを１次元にし，そのスロット集で各スロットの出現頻度を計測　#文脈のみの文の削除後
        temp_slot_type = list(count_e.keys())
        temp_slot_type.pop(temp_slot_type.index("o"))
        self.slot_type = temp_slot_type
        print("文脈削除前\n",count_i)
        print("文脈削除後\n",count_e) #データ整理がうまくいっているかの確認

        if len(self.ori_slot_list) != len(self.ori_sentence_list):
            raise ValueError()

    def create_sentence_list(self):
        self.sentence_list = []
        self.slot_list = []
        for i in range(len(self.ori_sentence_list)):
            self.duplicate_sentence(self.ori_sentence_list[i], self.ori_slot_list[i], self.ori_label_num_list[i])
        
    def duplicate_sentence(self, sentence, slots, label_num): #スロットを複数持つ場合，スロット一つだけ残したセンテンスを複製
        # ラベル1と2の位置を把握して，slot名と番号を0やoに置き換える(取り合えずラベルは触らない．使う予定がないので)
        if len(sentence) != len(slots):
            print(sentence)
            print(slots)
            raise ValueError()

        for i in range(len(sentence)):
            if label_num[i] == 1: #ラベルBの位置のスロットの獲得
                self.sentence_list.append(copy(sentence))
                self.slot_list.append(slots[i])
                break
        return

    def create_sentences_dict(self):
        keys = self.slot_type
        values = [[] for i in range(len(keys))]
        
        for i in range(len(self.sentence_list)):
            this_slot = self.slot_list[i]
            index = keys.index(this_slot)
            values[index].append(self.sentence_list[i])

        self.sentence_dict = dict(zip(keys, values))

        return

    def return_something(self):
        
        return {"sentence_list":copy(self.sentence_list), "slot_list":copy(self.slot_list), \
            "sentence_dict":copy(self.sentence_dict)}



