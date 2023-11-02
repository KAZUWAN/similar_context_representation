import copy
import torch
import seaborn as sns
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
from transformers import BertTokenizer, BertModel



def pileup_attention(attention_w_tensor, tokens= None, layer_n= None, save= False, show= False) -> None:
    heads = attention_w_tensor

    piled_attn = torch.sum(heads, dim= 0)
    # print(f"piled attention:\n {piled_attn}")

    masked_piled_attn = copy.copy(piled_attn)
    masked_piled_attn = masked_piled_attn.detach().numpy()
    masked_piled_attn[:,0] = 0 # mask CLS -> any words
    masked_piled_attn[:,-1] = 0 # mask SEP -> any words    
    diag = np.zeros(masked_piled_attn.shape[0]) # diag -> 0
    np.fill_diagonal(masked_piled_attn, diag)
    # print(f'masked piled attention: \n{masked_piled_attn}')
    # print(f'shape[0]:{masked_piled_attn.shape[0]}')
    # print(tokens)


    # plt.rcParams['figure.subplot.bottom'] = 0.30
    fig, ax = plt.subplots(1, 1, figsize= (16,16))

    fig.suptitle(f'piled attention', fontsize= 20)
    fig.supxlabel(f'key', fontsize= 20)
    fig.supylabel(f'query', fontsize= 20)

    sns.heatmap(masked_piled_attn.copy(), ax = ax, cmap= 'OrRd', square=True)
    ax.set_title(f'layer {layer_n}', fontsize= 20)
    ax.set_xticks(np.asarray(list(range(masked_piled_attn.shape[0])))+0.5, tokens, rotation= 90, fontsize= 16)
    ax.set_yticks(np.asarray(list(range(masked_piled_attn.shape[0])))+0.5, tokens, rotation= 0, fontsize= 16)
    '''color bar のフォントサイズ変更'''
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize = 16)

    if save:
        # この実行ファイルのパスを取得
        filepath = os.path.dirname(os.path.abspath(__file__))

        # このファイルから画像を保存するフォルダへのパス
        now = datetime.datetime.now()
        save_file = f'figures/piled_attention_l{layer_n}_{now.year:>04}{now.month:>02}{now.day:>02}{now.hour:>02}{now.minute:>02}{now.second:>02}.png'

        # 結合
        filepath = os.path.join(filepath, save_file)

        plt.savefig(filepath, pad_inches=0.05)
    if show:
        plt.show()
    plt.close()
    

def most_attendingto_wordfreq(attention_w_tensor, tokens, layer_n, frag_mask= False, save= True, show= False) -> None:
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
        pass
        # もしマスクするなら12ヘッド全てのCLS,SEPへの注意を0にする   

    # assert False

    # 12ヘッド回繰り返して，各単語からの最も着目した単語のインデックスを取得
    # その回数を12ヘッド回通して記録
    # そのレイヤーで最も注目した単語をインデックス番号を付けて出力　（例）3(index_number)_
    # print("shapeshapeshape")
    freq_list[0,0] = 1
    # print(freq_list[0,0])
    # print(freq_list)
    for i in range(12): # roop attention head num 12
        attention_w = attention_w_tensor[i]
        for j in range(attention_w.shape[0]):
             index = torch.argmax(attention_w[j])
             freq_list[j, index] += 1
            #  print(freq_list)
    # print(freq_list)

    print(f"\n{layer_n} most attend to:")
    for i in range(attention_w.shape[0]):
        index = np.argmax(freq_list[i])
        print(f"layer{tokens[i]} most attnend to {index}_{tokens[index]}")


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
    

