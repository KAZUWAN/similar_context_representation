from ..config import *
import numpy as np
import copy

class BatchManager():
    def __init__(self, sentence_list, slot_list, sentence_list_by_keys, sentence_only_o) -> None:
         self.sentence_list = sentence_list
         self.slot_list = slot_list
         self.sentence_list_by_keys = sentence_list_by_keys
         self.sentnece_only_o = sentence_only_o
         self.anchor_sentence_list = None
         self.anchor_slot_list = None
         self.positive_sentence_list = None
         self.negative_sentence_list = None
         
         return

    def create_anchor_positive_negative(self):
        self.create_anchor_sentence_list()
        self.create_positive_sentence_list() #同じスロットの文
        self.create_negative_sentence_list() #違うスロットの文

        return


    def create_anchor_sentence_list(self):
        index_random = list(np.random.randint(0, len(self.sentence_list), size=BATCH_SIZE))
        self.anchor_sentence_list = [self.sentence_list[i] for i in index_random]
        self.anchor_slot_list = [self.slot_list[i] for i in index_random]
        
        return

    def create_positive_sentence_list(self): #とりあえず，同じスロットの文で置き換えるパターン
        self.positive_sentence_list = []
        for i in range(BATCH_SIZE):
            self.positive_sentence_list.append(self.get_sameslot_sentence(self.anchor_slot_list[i]))

        return 

    def get_sameslot_sentence(self, anchor_slot):
        sameslot_sentence_list = []
        if type(anchor_slot) is list:
            for i in anchor_slot:
                sameslot_sentence_list.extend(self.sentence_list_by_keys[i])
        else:
            sameslot_sentence_list = self.sentence_list_by_keys[anchor_slot]

        index_random = np.random.randint(0, len(sameslot_sentence_list), size=1)[0]
        positive_sentence = sameslot_sentence_list[index_random]

        return positive_sentence

    def create_negative_sentence_list(self):
        self.negative_sentence_list = []
        for i in range(BATCH_SIZE):
            # print(f"anchor sentence: {self.anchor_sentence_list[i]}")
            # print(f"anchor slots: {self.anchor_slot_list[i]}")
            differenceslot_sentence_list = self.get_difference_slot_sentence_list(self.anchor_slot_list[i])
            index_random = np.random.randint(0, len(differenceslot_sentence_list), size=1)[0]
            self.negative_sentence_list.append(differenceslot_sentence_list[index_random])

        return

    def get_difference_slot_sentence_list(self, anchor_slot):
        slot_types = copy.copy(list(self.sentence_list_by_keys.keys()))
        difference_slottypes = copy.copy(slot_types)
        if type(anchor_slot) is list:
            for i in anchor_slot:
                difference_slottypes.remove(i)
        else:
            difference_slottypes.remove(anchor_slot)
        # print(f"anchor slot: {anchor_slot}, difference slot: {difference_slottypes}")
        if anchor_slot in difference_slottypes:
            raise ValueError("error: making negative; using same slot")
        # if len(difference_slottypes) == 0:
        #     raise ValueError("difference slot types is NONE")
        negative_candidate_list = []
        if len(difference_slottypes) == 0:
            negative_candidate_list = self.sentnece_only_o
        else:
            for i in range(len(difference_slottypes)):
                negative_candidate_list.extend(self.sentence_list_by_keys[difference_slottypes[i]])
            
        return negative_candidate_list

    def return_batch_data(self):
        return {"anchor_slot_list":copy.copy(self.anchor_slot_list), "anchor_sentence_list":copy.copy(self.anchor_sentence_list),\
             "positive_sentence_list":copy.copy(self.positive_sentence_list), "negative_sentence_list":copy.copy(self.negative_sentence_list)}
        


        



    
    

