import cv2
import numpy as np
import os

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

def crop_images(image, label, box, crop_range, n):
    """
    指定された範囲内で、画像とラベルデータをクロップする関数。

    Parameters:
    - image (numpy array): 画像データ (H, W)
    - label (numpy array): ラベルデータ (H, W)
    - box (tuple): クロップする範囲 (x_min, y_min, x_max, y_max)
    - crop_range (int): クロップする領域の大きさ (range x range)
    - n (int): クロップする画像の数

    Returns:
    - cropped_images (list): クロップされた画像のリスト
    - cropped_labels (list): クロップされたラベルデータのリスト
    """
    x_min, y_min, x_max, y_max = box
    cropped_images = []
    cropped_labels = []

    # boxの範囲内でランダムにクロップ位置を決定
    for _ in range(n):
        # x_min, y_minの範囲内でクロップ開始点を決定
        x_start = np.random.randint(x_min, x_max - crop_range)
        y_start = np.random.randint(y_min, y_max - crop_range)

        # 画像とラベルを同じ範囲でクロップ
        cropped_image = image[y_start:y_start + crop_range, x_start:x_start + crop_range]
        cropped_label = label[y_start:y_start + crop_range, x_start:x_start + crop_range]

        cropped_images.append(cropped_image)
        cropped_labels.append(cropped_label)

    return cropped_images, cropped_labels

# 画像とラベルを読み込む例
image_path = 'input_image.raw'
label_path = 'label_image.png'

# 画像サイズの指定（raw画像の高さと幅）
height, width = 1024, 1024  # 例として1024x1024サイズの画像と仮定

# raw画像を読み込む
image = load_raw_image(image_path, height, width)

# ラベル画像を読み込む（ラベルは通常のpngファイルと仮定）
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # ラベルデータ (グレースケール画像)

# クロップする範囲(box)とクロップサイズ(range)および数(n)の指定
box = (50, 50, 200, 200)  # クロップ範囲
crop_range = 64  # クロップサイズ (64x64ピクセル)
n = 10  # クロップ数

# クロップ実行
cropped_images, cropped_labels = crop_images(image, label, box, crop_range, n)

# クロップされた画像とラベルを保存
output_dir = 'cropped_outputs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, (cropped_image, cropped_label) in enumerate(zip(cropped_images, cropped_labels)):
    cv2.imwrite(f'{output_dir}/cropped_image_{i}.png', cropped_image)
    cv2.imwrite(f'{output_dir}/cropped_label_{i}.png', cropped_label)

print(f'{n} 個の画像とラベルが {output_dir} に保存されました')

