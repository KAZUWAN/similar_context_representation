import collections

class Functions01():
    def __init__(self) -> None:
        return None

    def remove_only_o(dataset): # 文脈語のみの文の削除 #ついでにスロットの種類とかも獲得する
        tokens = dataset["tokens"]
        labels = dataset["labels"]
        labels_num = dataset["labels_num"]
        slots = dataset["slots"]
        count_i = collections.Counter(sum (slots,[])) #slotsリストを１次元にし，そのスロット集で各スロットの出現頻度を計測 #文脈のみの文の削除前
        r = len(labels_num) #センテンス数
        for i in range(len(tokens)): #データ数の分だけ確認する
            if labels_num[r-1-i].count(1) == 0: #ラベル番号1  無いとき取り除く．スロットを持つ場合必ずラベルBをもつから #後ろの番号から調べる
                tokens.pop(r-1-i)
                labels.pop(r-1-i)
                labels_num.pop(r-1-i)
                slots.pop(r-1-i)
        count_e = collections.Counter(sum (slots,[])) #slotsリストを１次元にし，そのスロット集で各スロットの出現頻度を計測　#文脈のみの文の削除後
        print("文脈削除前\n",count_i)
        print("文脈削除後\n",count_e) #データ整理がうまくいっているかの確認
        return count_e.keys() #各スロット名を返す


    #スロットを持つ単語をスロットごとにリストアップしておく
    def extract_value(slot_names, tokens, slots): #set関数で重複をなくせる
        # 各スロット名とslotsのスロット列が一致する位置を獲得し，その位置の単語列をバリューにしまっていく
        # スロット名のindex番号に対応するvaluesのindexのリストに収納していく
        keys = slot_names
        values = [[]*len(keys)]
        tokens = sum(tokens, [])
        slots = sum(slots, [])

        for i in range(len(values)):
            for i in range(len(tokens)):

            values[i] = list(set(values[i])) #単語集の重複をなくし，集合型をリストに直す

        value_list = dict(zip(keys, values))
    
    # def to_ids(): #id化, スペシャルトークンの付加，このタイミングでバッチサイズ内の最大長でパディング
    # def BERT(): #オリジナルBERTでの分散表現の獲得