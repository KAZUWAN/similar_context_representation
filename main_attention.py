
import os
import torch
from transformers import BertTokenizer, BertModel
from src import config
from src.util import load_data
from src.util import make_datacollection02
import visualize_attention.mean_attentionw
import visualize_attention.attention_visualize02
from src.util import add_special_token


if __name__ == '__main__':
    
    input_path_tr = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), config.DATA_TRAIN_PATH))
    train_dataset = load_data.load_json_data(input_path_tr)

    creater = make_datacollection02.CreateOriginal(train_dataset)
    creater.remove_only_o()
    creater.create_sentence_list()
    creater.create_sentences_dict()
    all_sentences = creater.return_something()
    print(f"length all sentences : {len(all_sentences['sentence_list'])}")
    print(all_sentences['sentence_list'][0])
    print(all_sentences['sentence_list'][2])
    print(all_sentences['sentence_list'][4])

    sentences_list = all_sentences['sentence_list']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    # visualize_attention.mean_attentionw.mean_attention_w_by_hiddenlayer(sentences_list, bert_model= bert_model, tokenizer= tokenizer)

    length_list = [len(length) for length in all_sentences]
    max_length = max(length_list)

    sentences_add_token_dic = add_special_token.add_specialtokens(sentences_list, max_length, plus_token= True, padding= False)

    sentences_add_token = sentences_add_token_dic['specialtokens_added']
    attention_mask_list = sentences_add_token_dic['attention_mask_list']
    input_ids_list = [tokenizer.convert_tokens_to_ids(k) for k in sentences_add_token]

    input_ids = torch.tensor(input_ids_list[0]).unsqueeze(dim= 0)
    
    bert_out = bert_model(input_ids, output_hidden_states= True, output_attentions= True)
    attentions = bert_out['attentions']
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(dim= 0))
    for i, attention_n in enumerate(attentions):
        attention_w = attention_n[0]
        visualize_attention.attention_visualize02.show_attention_heatmap(attention_w_tensor= attention_w, tokens= tokens, layer_n=i, save = True, show= True)



    