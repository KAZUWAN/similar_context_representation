import json

def load_json_data(path): #loading dataset
    json_open = open(path,"r")
    json_load = json.load(json_open)
    json_open.close()
    
    data_tokens=[]
    data_labels=[]
    data_labels_num=[]
    data_slots = []
    
    for i in range(len(json_load)):
        for j in range(len(json_load[i]["turns"])):
            # 単語列の取得
            temp_tokens=json_load[i]["turns"][j]["user_utterance"]["tokens"]
            # ラベルの付与
            temp_labels=["o"]*len(temp_tokens) # label "o"
            temp_labels_num=[0]*len(temp_tokens) # label number 0
            # スロット名の獲得 文脈語なのでラベルo
            temp_slots = temp_labels.copy()
            
            have_slot = json_load[i]["turns"][j]["user_utterance"]["slots"] # slotを読み込み
            if len(have_slot)>=1: # have slot-value?
                for l in range(len(have_slot)): # 持っているスロット分だけ繰り返す
                    slot = have_slot[l]["slot"] # 手前のスロットから取得
                    temp_labels[have_slot[l]["start"]]="B" # label "B"
                    temp_labels_num[have_slot[l]["start"]]=1 # label number 1
                    temp_slots[have_slot[l]["start"]] = slot
                    
                    range_i_slot_start = have_slot[l]["start"]+1
                    range_i_slot_end = have_slot[l]["exclusive_end"]
                    len_i_slot = range_i_slot_end-range_i_slot_start
                    # B以降，end以前のスロットにlabel "i"
                    temp_labels[range_i_slot_start:range_i_slot_end]=["I"]*(len_i_slot)
                    # B以降，end以前のスロットにラベル2
                    temp_labels_num[range_i_slot_start:range_i_slot_end]=[2]*(len_i_slot)
                    # B以降，end以前のスロットを取得
                    temp_slots[range_i_slot_start:range_i_slot_end] = [slot]*(len_i_slot)
            if len(temp_tokens) != len(temp_slots):
                raise ValueError()
            data_tokens.append(temp_tokens) # tokensリストに収納
            data_labels.append(temp_labels) # ラベル列リストに収納
            data_labels_num.append(temp_labels_num) # ラベル番号列リストに収納
            data_slots.append(temp_slots) # slot列リストに収納
    for i in range(len(data_tokens)):
        if len(data_tokens) != len(data_slots):
            raise ValueError()
        if len(data_tokens[i]) != len(data_slots[i]):
            raise ValueError()
    return {"tokens":data_tokens,"labels":data_labels,"labels_num":data_labels_num, "slots":data_slots}