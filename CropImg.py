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

def resize_to_label(image, label):
    """
    画像のサイズをラベルと同じサイズにリサイズする関数。
    
    Parameters:
    - image (numpy array): リサイズする画像
    - label (numpy array): リサイズ先のサイズを持つラベル
    
    Returns:
    - resized_image (numpy array): ラベルと同じサイズにリサイズされた画像
    """
    label_height, label_width = label.shape[:2]
    resized_image = cv2.resize(image, (label_width, label_height))  # ラベルと同じサイズにリサイズ
    return resized_image

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
    print("Starting crop_images function...")
    
    for i in range(n):
        print(f"Cropping iteration {i+1}")
        # クロップ範囲が画像サイズ内に収まっているか確認
        if (x_max - x_min < crop_range) or (y_max - y_min < crop_range):
            raise ValueError("クロップ範囲が画像サイズを超えています。crop_rangeを小さくするか、boxの範囲を変更してください。")

        x_start = np.random.randint(x_min, x_max - crop_range)
        y_start = np.random.randint(y_min, y_max - crop_range)
        print(f"x_start: {x_start}, y_start: {y_start}")
        cropped_image = image[y_start:y_start + crop_range, x_start:x_start + crop_range]
        cropped_label = label[y_start:y_start + crop_range, x_start:x_start + crop_range]
         # クロップ結果が空でないか確認
        if cropped_image.size == 0 or cropped_label.size == 0:
            raise ValueError(f"クロップされた画像またはラベルが空です。boxの範囲や画像サイズを確認してください。")

        cropped_images.append(cropped_image)
        cropped_labels.append(cropped_label)
    print("Cropping complete, returning results.")
    return cropped_images, cropped_labels

# 画像とラベルを読み込む例
image_path = 'data/1ex/2400_SC.raw' #2400で実験
label_path = 'data/label/ps_center/2400.png'

# 画像サイズの指定（raw画像の高さと幅）
# height, width = 100, 460  # 460x100サイズの画像(gtのプロパティ参照)

# raw画像を読み込む
image = load_raw_image(image_path)

# ラベル画像を読み込む（ラベルは通常のpngファイルと仮定）
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # ラベルデータ (グレースケール画像)

# raw画像をラベルと同じサイズにリサイズ
image_resized = resize_to_label(image, label)

# クロップする範囲(box)とクロップサイズ(range)および数(n)の指定
box = (0, 0, 460, 100)  # クロップ範囲
crop_range = 25  # クロップサイズ (25x25ピクセル)
n = 10  # クロップ数

# クロップ実行
cropped_images, cropped_labels = crop_images(image_resized, label, box, crop_range, n)

# クロップされた画像とラベルを保存
output_dir = 'cropped_outputs/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, (cropped_image, cropped_label) in enumerate(zip(cropped_images, cropped_labels)):
    cv2.imwrite(f'{output_dir}/cropped_image_{i}.png', cropped_image)
    cv2.imwrite(f'{output_dir}/cropped_label_{i}.png', cropped_label)

print(f'{n} 個の画像とラベルが {output_dir} に保存されました')

