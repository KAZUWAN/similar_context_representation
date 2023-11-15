import copy
import torch
import seaborn as sns
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertModel
import time



def pileup_attention(attention_w_tensor:torch.tensor, tokens= None, layer_n= None, show= False, save= False,
                      mask_special_token=True, mask_diag= True, get_pileup_attention= False, return_path= False) -> None or np.array or list:
   
    filepath = None
    heads = attention_w_tensor
    piled_attn = torch.sum(heads, dim= 0)
    # print(f"piled attention:\n {piled_attn}")
    masked_piled_attn = copy.copy(piled_attn)
    masked_piled_attn = masked_piled_attn.detach().numpy()

    if mask_special_token:
        masked_piled_attn[:,0] = 0 # any words -> mask CLS
        masked_piled_attn[:,-1] = 0 # any words -> mask SEP 
        
    if mask_diag:   
        diag = np.zeros(masked_piled_attn.shape[0]) 
        np.fill_diagonal(masked_piled_attn, diag) # diag -> 0
    # print(f'masked piled attention: \n{masked_piled_attn}')
    # print(f'shape[0]:{masked_piled_attn.shape[0]}')
    # print(tokens)

    if save or show:
        # plt.rcParams['figure.subplot.bottom'] = 0.30
        fig1, ax1 = plt.subplots(1, 1, figsize= (16,16))
        fig1.suptitle(f'piled attention', fontsize= 20)
        fig1.supxlabel(f'key', fontsize= 20)
        fig1.supylabel(f'query', fontsize= 20)
        sns.heatmap(masked_piled_attn.copy(), ax = ax1, cmap= 'OrRd', square=True)
        ax1.set_title(f'layer {layer_n}', fontsize= 20)
        ax1.set_xticks(np.asarray(list(range(masked_piled_attn.shape[0])))+0.5, tokens, rotation= 90, fontsize= 16)
        ax1.set_yticks(np.asarray(list(range(masked_piled_attn.shape[0])))+0.5, tokens, rotation= 0, fontsize= 16)
        '''color bar のフォントサイズ変更'''
        cbar = ax1.collections[0].colorbar
        cbar.ax.tick_params(labelsize = 16)

        if show:
            fig1.show()

        if save:
            # この実行ファイルのパスを取得
            filepath = os.path.dirname(os.path.abspath(__file__))
            # このファイルから画像を保存するフォルダへのパス
            now = datetime.datetime.now()
            save_file = f'figures/piled_attention_l{layer_n}_{now.year:>04}{now.month:>02}{now.day:>02}{now.hour:>02}{now.minute:>02}{now.second:>02}.png'
            # 結合
            filepath = os.path.join(filepath, save_file)
            fig1.savefig(filepath, pad_inches=0.05)
            time.sleep(1.0)  
        plt.close(fig1)

    if get_pileup_attention and return_path:
        return masked_piled_attn, filepath
    
    elif get_pileup_attention:
        return masked_piled_attn
    
    elif return_path:
        return filepath
    

def most_attendingto_wordfreq(attention_w_tensor, tokens, layer_n, frag_mask= False) -> None:
    # レイヤーごとにアテンションを受け取り，各ヘッドの各行で最も着目している単語を記録していく
    # cls, sepを抜くか抜かないかを選択にするか，両方出すのが良い
    tokens = tokens
    # print(tokens)
    freq_list = np.zeros(attention_w_tensor[0].shape)
    # print(freq_list.shape)


    if frag_mask: # CLS, SEP をmaskするとき
        attention_w_tensor = copy.copy(attention_w_tensor)
        attention_w_tensor = attention_w_tensor.detach().numpy()
        attention_w_tensor[:, :, 0] = 0 # mask CLS -> any words
        attention_w_tensor[:, :, -1] = 0 # mask SEP -> any words
        attention_w_tensor = torch.from_numpy(attention_w_tensor.astype(np.float32)).clone()    
        # もしマスクするなら12ヘッド全てのCLS,SEPへの注意を0にする   

    # assert False

    # 12ヘッド回繰り返して，各単語からの最も着目した単語のインデックスを取得
    # その回数を12ヘッド回通して記録
    # そのレイヤーで最も注目した単語をインデックス番号を付けて出力　（例）3(index_number)_
    # print("shapeshapeshape")
    # freq_list[0,0] = 1
    # print(freq_list[0,0])
    # print(freq_list)
    for i in range(12): # roop attention head num 12
        attention_w = attention_w_tensor[i]

        for j in range(attention_w.shape[0]):
             index = torch.argmax(attention_w[j])
             freq_list[j, index] += 1
            #  print(freq_list)
    # print(freq_list)

    print(f"\nlayer{layer_n} most attend to:")
    for i in range(attention_w.shape[0]):
        index = np.argmax(freq_list[i])
        print(f"{tokens[i]} most attnend to {index}_{tokens[index]}")


def plieup_attentionw_graph(attention_w_tensors:tuple, tokens, show= False, save= False, mask_special_token=True, mask_diag= True, upto_n_layer=11):
    # 間違えた，一つの層内で，各単語がどの単語に着目しているかを見たかったのに，なんかよくわからんもの出てきた
    if (show == False) and (save == False):
        print(f"no operation")
        return
    else:
        pass

    cm = plt.get_cmap('Blues')
    # make figure
    fig1, ax1 = plt.subplots(1, 1, figsize=(32,16))

    for layer_n, attention_w in enumerate(attention_w_tensors):

        # legend付けないとな
        attn_w = pileup_attention(attention_w_tensor=attention_w.squeeze(dim=0), tokens=tokens, layer_n=layer_n, show=False, save=False, mask_special_token=mask_special_token, mask_diag=mask_diag, get_pileup_attention=True)
        total_attention_by_layer = attn_w.sum(axis=0)
        # print(f'attn_w.shape: {attn_w.shape}')
        # print(f"attn_w.sum(axis=0): {attn_w.sum(axis=0)}")
        ax1.plot(list(range(len(tokens))), total_attention_by_layer, color=cm(layer_n/len(attention_w_tensors)), label=f"Layer{layer_n}")

        if layer_n == 0:
            if mask_special_token & mask_diag:
                ax1.set_title(f'pileup attention weight  *masking current,SEP,CLS')
            elif mask_special_token:
                ax1.set_title(f'pileup attention weight  *masking SEP,CLS')
            elif mask_diag:
                ax1.set_title(f'pileup attention weight  *masking current')
            else:
                ax1.set_title(f'pileup attention weight')

            ax1.set_xlabel(f'Query word')
            ax1.set_ylabel(f'pileup attention weight')
            ax1.set_xticks(list(range(len(tokens))), tokens,  fontsize=16, rotation=90)
            ax1.set_ylim(0, 20)

            if layer_n == upto_n_layer:
                break

    ax1.legend(loc='upper left')

    if show:
        plt.show()
    
    if save:
        filepath = os.path.dirname(os.path.abspath(__file__))
        now = datetime.datetime.now()
        save_file = f'figures/pileupgraph_{now.year:>04}{now.month:>02}{now.day:>02}{now.hour:>02}{now.minute:>02}{now.second:>02}.png'
        filepath = os.path.join(filepath, save_file)
        fig1.savefig(filepath, pad_inches=0.05) 
        time.sleep(1.0) 
    plt.close(fig1)
    


if __name__ == "__main__":
    # a = torch.ones((1,2,2))
    # b = torch.full((1,2,2), fill_value= 2)
    # c = torch.full((1,2,2), fill_value= 3)
    # d = torch.cat([a,b,c])
    # print(d)
    # print(d.shape)
    # print(torch.sum(d, dim=0))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    text = "find a restaurant in orlando"

    tokens = tokenizer.tokenize(text)
    print(f'tokens: {tokens}')

    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)

    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))

    outputs = bert_model(input_ids, output_hidden_states= True, output_attentions= True)
    attentions = outputs['attentions']

    # print(f"attentions length: {len(attentions)}")
    # print({f"attentions[0].shape: {attentions[0].shape}"})
    # print({f"{attentions[0].squeeze(0).shape}"})
    # attention_heads_layer0 = attentions[0].squeeze(0)
    for i, attention_n in enumerate(attentions):
            attention_w = attention_n.squeeze(0)
            pileup_attention(attention_w_tensor= attention_w, tokens= tokens, layer_n=i, save = True, show= False)

    # print(f"heads0: {attention_heads_layer0[0]}")
    # print(f"heads1: {attention_heads_layer0[1]}")
    
    # heads = attention_heads_layer0[0:2]
    # print(f"heads.shape: {heads.shape}")
    # piled_attn = torch.sum(heads, dim= 0)
    # print(f"piled attention:\n {piled_attn}")

    # masked_piled_attn = copy.copy(piled_attn)
    # masked_piled_attn = masked_piled_attn.detach().numpy()
    # masked_piled_attn[:,0] = 0 # mask CLS -> any words
    # masked_piled_attn[:,-1] = 0# mask SEP -> any words
    # # 対角成分を0にする↓
    # print(f"shape: {masked_piled_attn.shape}")
    # diag = np.zeros(masked_piled_attn.shape[0])
    # print(f"diag: {diag}")
    # np.fill_diagonal(masked_piled_attn, diag)
    
    # print(f'masked piled attention: \n{masked_piled_attn}')
    

