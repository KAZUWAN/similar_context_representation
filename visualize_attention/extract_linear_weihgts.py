from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import datetime



def extract_lweights(model_name, show= True, save= False):
    # BERTモデルのロード
    model = BertModel.from_pretrained(model_name)
    hidden_size = 768
    n = 12

    # 12層だけ繰り返す
    for layer_n in range(n):
        # BERTのアテンション層を取得
        attention_layer = model.encoder.layer[layer_n].attention

        # Queryについて
        # Queryの線形層の重みの抽出
        query_weights = attention_layer.self.query.weight
        query_bias = attention_layer.self.query.bias

        
        w_max = query_weights.max()
        w_min = query_weights.min()
        if abs(w_max) >= abs(w_min):
            vmax = w_max
            vmin = w_max * -1
        else:
            vmax = w_min * -1
            vmin = w_min

        fig, ax = plt.subplots(2, 2, figsize= (20,20))

        fig.suptitle(f'Layer {layer_n}', fontsize= 20)
        # fig.supxlabel(f'key', fontsize= 20)
        # fig.supylabel(f'query', fontsize= 20)

        sns.heatmap(query_weights.detach().numpy().copy(), ax = ax[0,0], cmap= 'BrBG', vmin= vmin, vmax= vmax, square=True)
        ax[0,0].set_title(f'Query weights', fontsize= 20)
        ax[0,0].set_xticks(np.asarray(list(range(0, hidden_size+1, 64))), list(range(0, hidden_size+1, 64)), rotation= 90, fontsize= 20)
        ax[0,0].set_yticks(np.asarray(list(range(0, hidden_size+1, 64))), list(range(0, hidden_size+1, 64)), rotation= 0, fontsize= 20)
    
        '''color bar のフォントサイズ変更'''
        cbar = ax[0,0].collections[0].colorbar
        cbar.ax.tick_params(labelsize = 20)

        ax[1,0].plot(list(range(1, query_bias.shape[0]+1, 1)), query_bias.detach().numpy().copy())
        ax[1,0].set_title(f'Query bias', fontsize= 20)
        ax[1,0].set_xticks(np.asarray(list(range(0, hidden_size+1, 64))), list(range(0, hidden_size+1, 64)), rotation= 90, fontsize= 20)
        ax[1,0].tick_params(labelsize= 20, axis = 'y')

        # Keyについて
        # Keyの線形層の重みの抽出
        key_weights = attention_layer.self.key.weight
        key_bias = attention_layer.self.key.bias

        sns.heatmap(key_weights.detach().numpy().copy(), ax = ax[0,1], cmap= 'BrBG', vmin= vmin, vmax= vmax, square=True)
        ax[0,1].set_title(f'Key weights', fontsize= 20)
        ax[0,1].set_xticks(np.asarray(list(range(0, hidden_size+1, 64))), list(range(0, hidden_size+1, 64)), rotation= 90, fontsize= 20)
        ax[0,1].set_yticks(np.asarray(list(range(0, hidden_size+1, 64))), list(range(0, hidden_size+1, 64)), rotation= 0, fontsize= 20)
    
        '''color bar のフォントサイズ変更'''
        cbar = ax[0,1].collections[0].colorbar
        cbar.ax.tick_params(labelsize = 20)

        ax[1,1].plot(list(range(1, key_bias.shape[0]+1, 1)), key_bias.detach().numpy().copy())
        ax[1,1].set_title(f'Key bias', fontsize= 20)
        ax[1,1].set_xticks(np.asarray(list(range(0, hidden_size+1, 64))), list(range(0, hidden_size+1, 64)), rotation= 90, fontsize= 20)
        ax[1,1].tick_params(labelsize= 20, axis= 'y')
        

        if save:
            # この実行ファイルのパスを取得
            filepath = os.path.dirname(os.path.abspath(__file__))

            # このファイルから画像を保存するフォルダへのパス
            now = datetime.datetime.now()
            save_file = f'figures/qwbkwb_L{layer_n}_{now.year:>04}{now.month:>02}{now.day:>02}{now.hour:>02}{now.minute:>02}{now.second:>02}.png'

            # 結合
            filepath = os.path.join(filepath, save_file)

            plt.savefig(filepath, pad_inches=0.05)
        if show:
            plt.show()

        plt.close()


def inner_pweight(model_name, show= True, save= False):
    # BERTモデルのロード
    model = BertModel.from_pretrained(model_name)
    hidden_size = 768
    n = 12
    hn = 12
        
    # 12層だけ繰り返す
    for layer_n in range(n):
        # BERTのアテンション層を取得
        attention_layer = model.encoder.layer[layer_n].attention
        # Queryの線形層の重みの抽出
        query_weights = attention_layer.self.query.weight
        # Keyの線形層の重みの抽出
        key_weights = attention_layer.self.key.weight

        fig, ax = plt.subplots(1, 12, figsize= (50,4))
        fig.suptitle(f'Layer {layer_n}: Wqk [innerp weight]', fontsize= 20)

        # ---- ここで，12ヘッドごとに内積の計算 ----

        for head_n in range(hn):
            # 内積計算
            # 計算の便宜上，Query weight は転置してQueryに掛けられる
            temp_query_weights = torch.transpose(query_weights[head_n*64: (head_n+1)*64], 0, 1)
            # Keyも同様に転置してかけられるが，この後Query weightに対して転置して書けるので転置の転置でそのままでよい
            temp_key_weights = key_weights[head_n*64: (head_n+1)*64]

            innerp_weights = torch.matmul(temp_query_weights, temp_key_weights)

            w_max = innerp_weights.max()
            w_min = innerp_weights.min()
            if abs(w_max) >= abs(w_min):
                vmax = w_max
                vmin = w_max * -1
            else:
                vmax = w_min * -1
                vmin = w_min

            sns.heatmap(innerp_weights.detach().numpy().copy(), ax = ax[head_n], cmap= 'BrBG', vmin= vmin, vmax= vmax, square=True)
            ax[head_n].set_title(f'head {head_n}', fontsize= 20)
            ax[head_n].set_xticks(np.asarray(list(range(0, hidden_size+1, 64))), list(range(0, hidden_size+1, 64)), rotation= 90, fontsize= 8)
            ax[head_n].set_yticks(np.asarray(list(range(0, hidden_size+1, 64))), list(range(0, hidden_size+1, 64)), rotation= 0, fontsize= 8)
            


            '''color bar のフォントサイズ変更'''
            # cbar = ax[head_n].collections[0].colorbar
            # cbar.ax.tick_params(labelsize = 16)

        if save:
            # この実行ファイルのパスを取得
            filepath = os.path.dirname(os.path.abspath(__file__))

            # このファイルから画像を保存するフォルダへのパス
            now = datetime.datetime.now()
            save_file = f'figures/Wqk_L{layer_n}_{now.year:>04}{now.month:>02}{now.day:>02}{now.hour:>02}{now.minute:>02}{now.second:>02}.png'

            # 結合
            filepath = os.path.join(filepath, save_file)

            plt.savefig(filepath, pad_inches=0.05)
        if show:
            plt.show()
        plt.close()
        


if __name__ == "__main__":


    extract_lweights(model_name="bert-base-uncased", show= False, save= True)
    # inner_pweight(model_name="bert-base-uncased", show= False, save= True)
    

    
