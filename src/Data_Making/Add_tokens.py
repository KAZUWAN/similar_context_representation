


def add_tokens(data_tokens,data_labels,data_labels_num, data_slots, padding=True):
    max_length=max([len(i) for i in data_tokens])
    attention_mask=[]
    for i in range(len(data_tokens)):
        data_tokens[i].append("[SEP]")
        data_labels[i].append("o")
        data_labels_num[i].append(0)
        data_slots[i].append('o')

        data_tokens[i].insert(0,"[CLS]")
        data_labels[i].insert(0,"o")
        data_labels_num[i].insert(0,0)
        data_slots[i].insert(0,"o")
        attention_mask.append([1]*len(data_tokens[i]))

        if padding == True & len(data_tokens[i])<max_length+2:
            temp_pad=["[PAD]"]*(max_length+2-len(data_tokens[i]))
            temp_0=[0]*(max_length+2-len(data_tokens[i]))
            temp_3=[3]*(max_length+2-len(data_tokens[i]))

            data_tokens[i].extend(temp_pad)
            data_labels[i].extend(temp_pad)
            data_slots[i].extend(temp_pad)
            data_labels_num[i].extend(temp_3)
            attention_mask[i].extend(temp_0)


    return {"tokens_add":data_tokens,"labels_add":data_labels,"labels_num_add":data_labels_num, 'slots_add':data_slots, "attention_mask":attention_mask}
