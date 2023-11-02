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
    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231027210525.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231027210525.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231027210525.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231027210526.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231027210526.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231027210526.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231027210526.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231027210527.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231027210527.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231027210527.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231027210528.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231027210528.png"]

    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231027210521.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231027210521.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231027210521.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231027210522.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231027210522.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231027210522.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231027210523.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231027210523.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231027210523.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231027210524.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231027210524.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231027210524.png",]



    # image_paths = ["./",
    #                "./",
    #                "./",
    #                "./",
    #                "./",
    #                "./",
    #                "./",
    #                "./",
    #                "./",
    #                "./",
    #                "./",
    #                "./",]

    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231027210510.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231027210511.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231027210511.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231027210511.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231027210512.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231027210512.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231027210512.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231027210513.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231027210513.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231027210513.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231027210514.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231027210514.png",]

    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231027210507.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231027210507.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231027210508.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231027210508.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231027210508.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231027210508.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231027210509.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231027210509.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231027210509.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231027210510.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231027210510.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231027210510.png",]

    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231027210504.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231027210504.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231027210505.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231027210505.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231027210505.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231027210505.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231027210506.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231027210506.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231027210506.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231027210506.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231027210507.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231027210507.png",]

    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231027210501.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231027210501.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231027210501.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231027210502.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231027210502.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231027210502.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231027210503.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231027210503.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231027210503.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231027210503.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231027210504.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231027210504.png",]

    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231027210458.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231027210458.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231027210458.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231027210459.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231027210459.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231027210459.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231027210459.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231027210500.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231027210500.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231027210500.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231027210500.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231027210501.png",]

    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231027210455.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231027210455.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231027210455.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231027210456.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231027210456.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231027210456.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231027210456.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231027210457.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231027210457.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231027210457.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231027210457.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231027210458.png",]
    
    # image_paths = ["./visualize_attention/figures/piled_attention_l0_20231031140449.png",
    #                "./visualize_attention/figures/piled_attention_l1_20231031140449.png",
    #                "./visualize_attention/figures/piled_attention_l2_20231031140449.png",
    #                "./visualize_attention/figures/piled_attention_l3_20231031140450.png",
    #                "./visualize_attention/figures/piled_attention_l4_20231031140450.png",
    #                "./visualize_attention/figures/piled_attention_l5_20231031140450.png",
    #                "./visualize_attention/figures/piled_attention_l6_20231031140450.png",
    #                "./visualize_attention/figures/piled_attention_l7_20231031140451.png",
    #                "./visualize_attention/figures/piled_attention_l8_20231031140451.png",
    #                "./visualize_attention/figures/piled_attention_l9_20231031140451.png",
    #                "./visualize_attention/figures/piled_attention_l10_20231031140451.png",
    #                "./visualize_attention/figures/piled_attention_l11_20231031140452.png",]

    image_paths = ["./visualize_attention/figures/piled_attention_l0_20231031151250.png",
                   "./visualize_attention/figures/piled_attention_l1_20231031151250.png",
                   "./visualize_attention/figures/piled_attention_l2_20231031151250.png",
                   "./visualize_attention/figures/piled_attention_l3_20231031151250.png",
                   "./visualize_attention/figures/piled_attention_l4_20231031151251.png",
                   "./visualize_attention/figures/piled_attention_l5_20231031151251.png",
                   "./visualize_attention/figures/piled_attention_l6_20231031151251.png",
                   "./visualize_attention/figures/piled_attention_l7_20231031151251.png",
                   "./visualize_attention/figures/piled_attention_l8_20231031151252.png",
                   "./visualize_attention/figures/piled_attention_l9_20231031151252.png",
                   "./visualize_attention/figures/piled_attention_l10_20231031151252.png",
                   "./visualize_attention/figures/piled_attention_l11_20231031151253.png",]
    
    concatenate_fig(image_paths= image_paths, save= True, show= False)



