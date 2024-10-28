import random
import os
import numpy as np
from PIL import Image

def load_raw_image(file_path, h=1344, w=1344):
    """
    .rawファイルを読み込み、指定されたサイズにリシェイプする関数。
    
    Parameters:
    - file_path (str): rawファイルのパス
    - h (int): 画像の高さ
    - w (int): 画像の幅
    
    Returns:
    - data (numpy array): 画像データ
    """
    with open(file_path, "rb") as f:
        rawdata = f.read()
        data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
    return data

def save_raw_image(data, file_path):
    """
    numpy配列を.raw形式で保存する関数。
    
    Parameters:
    - data (numpy array): 保存する画像データ
    - file_path (str): 保存先のファイルパス
    """
    data.tofile(file_path)

# ランダムなデータ拡張を行う関数
def random_augmentation(image):
    # ランダムに左右反転
    if random.random() > 0.5:
        image = np.fliplr(image)
    
    # ランダムに上下反転
    if random.random() > 0.5:
        image = np.flipud(image)
    
    # ランダムに拡大・縮小（90%～110%）
    scale = random.uniform(0.9, 1.1)
    h, w = image.shape
    new_size = (int(w * scale), int(h * scale))
    image = np.array(Image.fromarray(image).resize(new_size, Image.ANTIALIAS))
    
    # 画像サイズを元のサイズにリサイズ
    image = np.array(Image.fromarray(image).resize((w, h), Image.ANTIALIAS))
    
    return image

# 画像のデータ拡張を行う関数
def augment_raw_images(input_dir, n, output_dir, h=1344, w=1344):
    # 入力フォルダ内のすべてのrawファイルを取得
    input_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.raw')]
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(n):
        # フォルダ内からランダムに画像を選択
        img_path = random.choice(input_images)
        img = load_raw_image(img_path, h, w)
        
        # ランダムなデータ拡張を適用
        augmented_img = random_augmentation(img)
        
        # 画像を保存
        output_path = os.path.join(output_dir, f"augmented_image_{i+1}.raw")
        save_raw_image(augmented_img, output_path)

# コード実行
input_dir = "input_images_folder"  # 入力raw画像フォルダのパス
n = 10  # 生成する画像の枚数
output_dir = "augmented_images"  # 出力先フォルダ

augment_raw_images(input_dir, n, output_dir)
