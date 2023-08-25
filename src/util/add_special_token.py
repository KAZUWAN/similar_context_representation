import copy

def add_specialtokens(input_sentence_list, max_length, plus_token=True, padding=True):
    attention_mask_list=[]
    sentence_list = copy.copy(input_sentence_list)
    
    for i in range(len(sentence_list)): 
        # print(f"len sentence list:{len(sentence_list)}")
        if plus_token == True:
            if sentence_list[i][0] != "[CLS]":
                sentence_list[i].append("[SEP]")
                sentence_list[i].insert(0,"[CLS]")
        attention_mask_list.append([1]*len(sentence_list[i]))
        # print("sentence length:",len(sentence_list[i]))
        if padding == True & (len(sentence_list[i]) < (max_length+2)):
            # print("##  padding...  ##")
            temp_pad=["[PAD]"]*(max_length+2-len(sentence_list[i]))
            temp_0=[0]*(max_length+2-len(sentence_list[i]))
            sentence_list[i].extend(temp_pad)
            attention_mask_list[i].extend(temp_0)
        
        # print(sentence_list[i])
        # print(attention_mask_list[i])
        # print("sentence length:",len(sentence_list[i]))
        if plus_token == True & (len(sentence_list[i]) != max_length+2) & (padding):
            raise ValueError("padding miss")

    return {"specialtokens_added":sentence_list, "attention_mask_list":attention_mask_list}
