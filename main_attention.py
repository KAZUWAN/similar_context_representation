
import os
from transformers import BertTokenizer, BertModel
from src import config
from src.util import load_data
from src.util import make_datacollection02
import visualize_attention.mean_attentionw


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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    visualize_attention.mean_attentionw.mean_attention_w_by_hiddenlayer(all_sentences['sentence_list'], bert_model= bert_model, tokenizer= tokenizer)


    