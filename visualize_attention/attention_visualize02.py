import torch
from transformers import BertTokenizer, BertModel
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime


def show_attention_heatmap(attention_w_tensor, text, layer_n, tokenizer, save= True, show= False):
    fig, ax = plt.subplots(3, 4, figsize= (40,30))
    fig.subplots_adjust(wspace= 0.27, hspace= 0.05)

    fig.suptitle(f'attention_weight at layer{layer_n}', fontsize= 40)
    fig.supxlabel(f'key', fontsize= 40)
    fig.supylabel(f'query', fontsize= 40)
    for i in range(12): # roop attention head num 12
        attention_w = attention_w_tensor[i]
        sns.heatmap(attention_w.detach().numpy().copy(), ax = ax[i//4, i%4], cmap= 'OrRd', annot= True, vmin=0.0, vmax= 1.0, fmt='.2f', square=True, annot_kws={'fontsize':19})
        ax[i//4, i%4].set_title(f'head{i}', fontsize= 30)

        ax[i//4, i%4].set_xticks(np.asarray(list(range(len(attention_w))))+0.5, tokenizer.convert_ids_to_tokens(tokenizer.encode(text)), rotation= 45, fontsize= 19)
        ax[i//4, i%4].set_yticks(np.asarray(list(range(len(attention_w))))+0.5, tokenizer.convert_ids_to_tokens(tokenizer.encode(text)), rotation= 0, fontsize= 22)

        '''color bar のフォントサイズ変更'''
        cbar = ax[i//4, i%4].collections[0].colorbar
        cbar.ax.tick_params(labelsize = 19)

    if save:
        # この実行ファイルのパスを取得
        filepath = os.path.dirname(os.path.abspath(__file__))

        # このファイルから画像を保存するフォルダへのパス
        now = datetime.datetime.now()
        save_file = f'figures/attention_w_l{layer_n}_{now.year:>04}{now.month:>02}{now.day:>02}{now.hour:>02}{now.minute:>02}{now.second:>02}.png'

        # 結合
        filepath = os.path.join(filepath, save_file)

        plt.savefig(filepath)
    if show:
        plt.show()



if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # text = "look for a restaurant at mountain view"
    text = "find a restaurant in orlando"

    tokens = tokenizer.tokenize(text)
    print(f'tokens: {tokens}')

    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    print(f'input_ids": {input_ids}')

    something = bert_model(input_ids, output_hidden_states= True, output_attentions= True)

    print(something.keys())
    keys = list(something.keys())


    print(f'\noutput size :{len(something)}')
    print(f'1.{keys[0]};   {something[0].shape}')
    print(f'2.{keys[1]};   {something[1].shape}')
    print(f'3.{keys[2]};   {len(something[2])}')
    print(f'4.{keys[3]};   {len(something[3])}')
    print(f'something[2][0].shape :{something[2][0].shape}')
    print(f'something[3][0].shape :{something[3][0].shape}')


    # "hidden states[0]は最初のembedding 出力"
    # 'hidden states[1:]が隠れ層の出力'

    print(f'\n---- attention weight at fitst head at last layer ')
    print(something[3][0][0][0].shape)
    print(something[3][0][0][0])

    # layer_n = 0 # layer number from 0 to 12
    # attention_tensor = something['attentions'][layer_n] #layer number

    attentions = something['attentions']
    print(f'attentions len: {len(attentions)}')
    print(f'one of attention layer shape: {attentions[0].shape}; [bathc-size, layer_n, key_n, query_n]')


    # print(f'{something['attentions'].shape}')

    print(f'{text}')

    temp_list = []
    for i, attention_layer in enumerate(attentions):
        layer_n = i
        attention_tensor = attention_layer[0] # 0 is batch number
        # show_attention_heatmap(attention_tensor, text, layer_n, tokenizer, save= True, show= False) #batch number 0
        # assert False
        temp_attention = attention_tensor[i]
        # print(f'attention shape: \n{temp_attention}')
        temp_attention = temp_attention[:-1, :-1]
        # print(f'attention shape: \n{temp_attention}')
        print(f'attention tensor mean: {temp_attention.mean()}')
        temp_list.append(temp_attention.mean().detach().numpy())

    fig = plt.figure()
    plt.plot(list(range(len(temp_list))), temp_list)
    plt.show()



