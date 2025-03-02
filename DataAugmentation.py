import random
import os
import shutil
import numpy as np
import cv2

# RAW画像を読み込む関数
def load_raw_image(file_path, h, w):
    with open(file_path, "rb") as f:
        rawdata = f.read()
        data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
    return data

# RAW画像を保存する関数
def save_raw_image(data, file_path):
    with open(file_path, "wb") as f:
        f.write(data.tobytes())

# ランダムなデータ拡張を行う関数
def random_augmentation(image, label, h, w):
    augmented = False
    while not augmented:
        # 左右反転
        if random.random() > 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)
            print("左右反転を適用")
            augmented = True
        
        # 上下反転
        if random.random() > 0.5:
            image = np.flipud(image)
            label = np.flipud(label)
            print("上下反転を適用")
            augmented = True
        
        # 拡大・縮小
        if random.random() > 0.5:
            scale = random.uniform(0.5, 2.0)
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, new_size, interpolation=cv2.INTER_NEAREST)
            print(f"拡大・縮小を適用（スケール: {scale:.2f}）")
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            augmented = True

    return image, label

# 画像とラベルのデータ拡張を行う関数
def augment_images_and_labels(image_dir, label_dir, n, output_image_dir, output_label_dir, h, w):
    # 出力ディレクトリを初期化
    if os.path.exists(output_image_dir):
        shutil.rmtree(output_image_dir)
    if os.path.exists(output_label_dir):
        shutil.rmtree(output_label_dir)
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 入力フォルダ内の画像とラベルを取得
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.raw')])
    labels = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')])

    for img_path, label_path in zip(images, labels):
        img_name = os.path.splitext(os.path.basename(img_path))[0]  # 拡張子を除去
        label_name = os.path.splitext(os.path.basename(label_path))[0]  # 拡張子を除去

        # 元画像とラベルを読み込む
        img = load_raw_image(img_path, h, w)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        for i in range(n):
            # データ拡張を適用
            augmented_img, augmented_label = random_augmentation(img.copy(), label.copy(), h, w)

            # 加工後の画像とラベルの名前を生成
            img_output_name = f"{i+1}_{img_name}_augmented.raw"
            label_output_name = f"{i+1}_{label_name}_augmented.png"

            # 加工後の画像とラベルを保存
            img_output_path = os.path.join(output_image_dir, img_output_name)
            label_output_path = os.path.join(output_label_dir, label_output_name)

            save_raw_image(augmented_img, img_output_path)
            cv2.imwrite(label_output_path, augmented_label)

# trainとvalidationに対して処理を実行する関数
def augment_train_and_validation(base_dir, n, h, w):
    for subset in ["train", "validation"]:
        image_dir = os.path.join(base_dir, "image", subset)
        label_dir = os.path.join(base_dir, "label", subset)
        output_image_dir = os.path.join(base_dir, "augmented_images", subset)
        output_label_dir = os.path.join(base_dir, "augmented_labels", subset)

        augment_images_and_labels(image_dir, label_dir, n, output_image_dir, output_label_dir, h, w)

# 実行
base_dir = "data/third"  # ベースディレクトリ
n = 10  # 各画像ごとに生成する加工画像の枚数
h, w = 32, 32  # 元画像およびラベルの高さと幅

augment_train_and_validation(base_dir, n, h, w)

