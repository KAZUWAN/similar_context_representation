from ..config import *
import numpy as np
import copy

class BatchManager():
    def __init__(self, sentence_list, slot_list, sentence_list_by_keys) -> None:
         self.sentence_list = sentence_list
         self.slot_list = slot_list
         self.sentence_list_by_keys = sentence_list_by_keys
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
        sameslot_sentence_list = self.sentence_list_by_keys[anchor_slot]
        index_random = np.random.randint(0, len(sameslot_sentence_list), size=1)[0]
        positive_sentence = sameslot_sentence_list[index_random]

        return positive_sentence

    def create_negative_sentence_list(self):
        self.negative_sentence_list = []
        for i in range(BATCH_SIZE):
            differenceslot_sentence_list = self.get_difference_slot_sentence_list(self.anchor_slot_list[i])
            index_random = np.random.randint(0, len(differenceslot_sentence_list), size=1)[0]
            self.negative_sentence_list.append(differenceslot_sentence_list[index_random])

        return

    def get_difference_slot_sentence_list(self, anchor_slot):
        slot_types = list(self.sentence_list_by_keys.keys())
        anchor_slottype_index = slot_types.index(anchor_slot)
        difference_slottypes = slot_types[:anchor_slottype_index] + slot_types[anchor_slottype_index+1:]
        if anchor_slot in difference_slottypes:
            raise ValueError("error: making negative; using same slot")
        negative_candidate_list = []
        for i in range(len(difference_slottypes)):
            negative_candidate_list.extend(self.sentence_list_by_keys[difference_slottypes[i]])
            
        return negative_candidate_list

    def return_batch_data(self):
        return {"anchor_slot_list":copy.copy(self.anchor_slot_list), "anchor_sentence_list":copy.copy(self.anchor_sentence_list),\
             "positive_sentence_list":copy.copy(self.positive_sentence_list), "negative_sentence_list":copy.copy(self.negative_sentence_list)}
        


        



    
    

