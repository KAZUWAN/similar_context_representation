#  使用した評価データのうち，BERTの事前学習に含まれていないUNKラベルを抽出する
#  トークンが100（[UNK]）のものを探す
#  それに対して評価を行い，事前学習に含まれなかった単語に対する精度の確認をする

#  使用したデータのうち訓練データに含まれていない評価データ(not include)を抽出する
#  それに対して評価を行い，訓練データに含まれなかった単語に対する精度を確認する


#  一つの文の中でもunkと既知が含まれているときのための対処が必要
##  一時的な対処としては，unkを含む文を対象に，それ以外を評価データから取り除く
##  他には，unkを含む文を抽出，正解を照らし合わせるときの既知ラベルを文脈とする

#  これらの調査と同時に，どれくらいunkが含まれているか記録しておく
#  また，具体的にどんな単語かも出力しておくと，今後の研究で有益かも


#  UNKの抽出

import os
from src.config import *

def extract_UNK(MakeSemanticSentenceEmbedding, eval_data, eval_slot):
    path_log = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),logprint_path))
    extract_UNKsentence = []
    extract_UNKslot = [] #UNK以外は文脈語（'o'）にラベルを変更 slotリストになってるのでoかslot名になっている
    UNK_store = [] # UNKになった単語を一覧として保存
    input_ids_list = MakeSemanticSentenceEmbedding.bert_ids(eval_data)["input_ids_list"] # データのトークン化（評価文たちを入力に想定）
    # print(len(input_ids_list))
    for i in range(len(input_ids_list)):
        temp_unk_num = []
        # print(eval_data[i])
        # print(input_ids_list[i])
        # print(MakeSemanticSentenceEmbedding.tokenizer.convert_ids_to_tokens(input_ids_list[i]))
        
        for j in range(len(input_ids_list[i])): # トークン化された文ずつ100（unk）を確認
            if input_ids_list[i][j] == 100:
                UNK_store.append(eval_data[i][j]) # unkとなる単語を保管
                temp_unk_num.append(j) # unkの位置indexをメモ
        if len(temp_unk_num) >0:
            extract_UNKsentence.append(eval_data[i]) # unkを含む文の抽出
            temp_UNKslot = ["o"]*len(input_ids_list[i]) # 文の長さ
            for k in temp_unk_num:
                temp_UNKslot[k] = eval_slot[i][k] # メモした番号の位置のラベルを本来のラベルに置き換える
            # print(temp_UNKslot)
            extract_UNKslot.append(temp_UNKslot)
    with open(path_log, mode = "a") as logf:
        print(f"len(extract_UNKsentence): {len(extract_UNKsentence)}", file= logf)
        print(f"len(extract_UNKslot): {len(extract_UNKslot)}", file= logf)

    if len(extract_UNKsentence) != len(extract_UNKslot):
        ValueError("extract Error: a number of UNKsentence not eqaul a number of UNKslot sequence")    
    UNK_store = list(set(UNK_store)) # UNKとなる単語の保管庫の重複をなくす

    return extract_UNKsentence, extract_UNKslot, UNK_store
            

                
                




    # 既知かどうか確認 (# トークン化，１を探す) # 既知のデータはラベルをoとしてしまうことで，訓練データに含まれることを防ぐ
    # unkが含まれる場合，既知を文脈語に置き換える．unkが含まれない場合，その文に含まれる単語を訓練データに含まない（すべてのラベルをoにする
    

    # データ作成時にあらかじめunkを抽出するのが効率がよさそうだが，もともと作成する予定だったデータに対するプログラムの方が複雑さはなくなると思われる
    # knn の動作前に，unkを抽出するか聞くプログラムを作成するのがよさそう


#  not includeの抽出
def extract_notinclude(train_data, eval_data):
    return


## データ評価に使用するデータにはスペシャルトークンを付与していない可能性があり，訓練時の入力と同様に文頭と文末にトークンを付与したら少しは精度が変わるのでは