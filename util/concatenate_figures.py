import os
from PIL import Image
import datetime
# import numpy as np


def concatenate_fig(image_paths:list, save= False, show= False):

    # 画像を読み込む
    images = [Image.open(path) for path in image_paths]

    # 画像の幅と高さを取得
    width, height = images[0].size

    # 連結後の画像の幅と高さを計算
    new_width = width * len(images)
    new_height = height

    # 連結後の画像のための新しいImageオブジェクトを作成
    new_image = Image.new('RGB', (new_width, new_height))

    # 画像を横に連結
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += width

    # この実行ファイルのパスを取得
    filepath = os.path.dirname(os.path.abspath(__file__))

    # このファイルから画像を保存するフォルダへのパス
    now = datetime.datetime.now()
    save_file = f'figures/concatenate_{now.year:>04}{now.month:>02}{now.day:>02}{now.hour:>02}{now.minute:>02}{now.second:>02}.png'

    # 結合
    filepath = os.path.join(filepath, save_file)

    # 連結した画像を保存
    if save:
        new_image.save(filepath)

    # 画像を表示する場合（オプション）
    if show:
        new_image.show()



if __name__ == "__main__":

    # 画像ファイルのパスをリストで指定




    image_paths = ["./",
                   "./",
                   "./",
                   "./",
                   "./",
                   "./",
                   "./",
                   "./",
                   "./",
                   "./",
                   "./",
                   "./",]

    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231106233427.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231106233427.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231106233427.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231106233428.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231106233428.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231106233428.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231106233428.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231106233428.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231106233429.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231106233429.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231106233429.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231106233429.png",]

    
    concatenate_fig(image_paths= image_paths, save= True, show= False)



