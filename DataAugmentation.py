import random
import os
from PIL import Image, ImageFilter

# ランダムなデータ拡張を行う関数
def random_augmentation(image):
    # ランダムに左右反転
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # ランダムに上下反転
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # ランダムに拡大・縮小（90%～110%）
    scale = random.uniform(0.9, 1.1)
    width, height = image.size
    new_size = (int(width * scale), int(height * scale))
    image = image.resize(new_size, Image.ANTIALIAS)
    
    # ランダムにぼかしを追加
    if random.random() > 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

    return image

# 画像のデータ拡張を行う関数
def augment_images(input_images, n, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(n):
        # 画像をランダムに選択
        img_path = random.choice(input_images)
        img = Image.open(img_path)
        
        # ランダムなデータ拡張を適用
        augmented_img = random_augmentation(img)
        
        # 画像を保存
        augmented_img.save(os.path.join(output_dir, f"augmented_image_{i+1}.png"))

# コード実行
input_images = ["image1.png", "image2.png"]  # 2枚の入力画像のパス
n = 10  # 生成する画像の枚数
output_dir = "augmented_images"  # 出力先フォルダ

augment_images(input_images, n, output_dir)
