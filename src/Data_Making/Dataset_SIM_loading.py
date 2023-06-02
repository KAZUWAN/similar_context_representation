import json

# loading datasets of SIM

# get return with dictionary type
    # get word tokens with list
    # get labels with list
    # get labels number with list

def json_loading(path):
    json_open=open(path,"r")
    json_load=json.load(json_open)
    json_open.close()
    
    data_tokens=[]
    data_labels=[]
    data_labels_num=[]
    data_slots = []
    
    for i in range(len(json_load)):
        for j in range(len(json_load[i]["turns"])):

            temp_tokens=json_load[i]["turns"][j]["user_utterance"]["tokens"]

            temp_labels=["o"]*len(temp_tokens) 
            temp_labels_num=[0]*len(temp_tokens)
            temp_slots = temp_labels.copy()

            have_slot = json_load[i]["turns"][j]["user_utterance"]["slots"]
            if len(have_slot)>=1: #have slot-value?
                for l in range(len(have_slot)):
                    slot = have_slot[l]["slot"]

                    temp_labels[have_slot[l]["start"]]="B"
                    temp_labels_num[have_slot[l]["start"]]=1

                    temp_labels[have_slot[l]["start"]+1:have_slot[l]["exclusive_end"]]=["I"]*(have_slot[l]["exclusive_end"]-have_slot[l]["start"]-1)
                    temp_labels_num[have_slot[l]["start"]+1:have_slot[l]["exclusive_end"]]=[2]*(have_slot[l]["exclusive_end"]-have_slot[l]["start"]-1)

                    temp_slots[have_slot[l]["start"]:have_slot[l]["exclusive_end"]] = [slot]*(have_slot[l]["exclusive_end"]-have_slot[l]["start"]-1)
            data_tokens.append(temp_tokens)
            data_labels.append(temp_labels)
            data_labels_num.append(temp_labels_num)
            data_slots.append(temp_slots)
            
    return {"tokens":data_tokens,"labels":data_labels,"labels_num":data_labels_num, "slots":data_slots}