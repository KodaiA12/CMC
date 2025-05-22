import torch
import numpy as np
from PIL import Image
import os
import glob
import re
import cv2
import segmentation_models_pytorch as smp
from scipy.ndimage import rotate  # 回転用の関数


def load_raw_image(file_path, h, w):
    """
    RAW画像をnumpy配列として読み込む関数。
    """
    with open(file_path, "rb") as f:
        raw_data = f.read()
        image = np.frombuffer(raw_data, dtype=np.int16).reshape(h, w)
    return image


def split_and_merge_with_inference(image, model, device, grid_size=32):
    """
    大きな画像を分割してモデルで推論し、再結合する関数。
    """
    model.eval()  # 推論モードに設定
    h, w = image.shape

    # 画像をパディング
    y_surp, x_surp = h % grid_size, w % grid_size
    padded_image = np.pad(
        image, ((0, grid_size - y_surp), (0, grid_size - x_surp)), "constant"
    )

    predicted_patches = []

    ph, pw = padded_image.shape
    for i in range(0, ph, grid_size):
        for j in range(0, pw, grid_size):
            patch = padded_image[i : i + grid_size, j : j + grid_size]
            normalized_patch = (patch.astype('float32') - patch.min()) / (patch.max() - patch.min())
            input_patch = torch.from_numpy(normalized_patch).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_patch)
                prediction = output.squeeze().cpu().numpy()
            predicted_patch = (prediction > 0.5).astype(np.uint8) * 255
            predicted_patches.append(predicted_patch)

    # 再結合
    reconstructed_image = np.zeros_like(padded_image, dtype=np.uint8)
    idx = 0
    for i in range(0, ph, grid_size):
        for j in range(0, pw, grid_size):
            reconstructed_image[i : i + grid_size, j : j + grid_size] = predicted_patches[idx]
            idx += 1
    reconstructed_image = reconstructed_image[:h, :w]
    return reconstructed_image


import cv2

def evaluate_predictions(result_file_path, gt_file_path, save_path):
    """
    推論結果とラベルデータの比較（TP, TN, FP, FN）から精度を計算し、結果を保存する関数。
    
    Parameters:
    - result_file_path (str): 推論結果の画像ファイルパス（PNG形式）
    - gt_file_path (str): 正解ラベル画像のファイルパス（PNG形式）
    - save_path (str): 評価結果を保存するテキストファイルのパス
    """
    # 画像の読み込み
    img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_file_path, cv2.IMREAD_GRAYSCALE)

    if img is None or gt is None:
        raise FileNotFoundError("推論画像または正解画像が見つかりません。パスを確認してください。")

    # TP, FP, TN, FN の初期化
    TP, FP, TN, FN = 0, 0, 0, 0

    # 画像の高さと幅を取得
    Y, X = gt.shape

    # ピクセルごとに比較
    for y in range(Y):
        for x in range(X):
            if img[y, x] == 255 and gt[y, x] == 255:
                TP += 1  # 真陽性
            elif img[y, x] == 0 and gt[y, x] == 0:
                TN += 1  # 真陰性
            elif img[y, x] == 0 and gt[y, x] == 255:
                FN += 1  # 偽陰性
            elif img[y, x] == 255 and gt[y, x] == 0:
                FP += 1  # 偽陽性

    # 精度、再現率、適合率、F値の計算
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f_measure = (2 * recall * precision) / (recall + precision) if (recall + precision) != 0 else 0

    # 結果を文字列に整形
    result_text = (
        f"推論結果画像: {result_file_path}\n"
        f"正解ラベル画像: {gt_file_path}\n"
        f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
        f"精度: {accuracy:.4f}, 再現率: {recall:.4f}, 適合率: {precision:.4f}, F値: {f_measure:.4f}\n"
    )

    # 結果を出力
    print(result_text)

    # 結果をファイルに保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ディレクトリが存在しない場合は作成
    with open(save_path, 'w') as f:
        f.write(result_text)

    print(f"評価結果を {save_path} に保存しました。")

def rotate_raw(file_path, save_dir, w, h, deg=15):
    # データの読み込み
    with open(file_path, "rb") as f:
        name = os.path.basename(file_path)
        rawdata = f.read()
        data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, name), "wb") as f:
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        rotated_data = cv2.warpAffine(data, mat, (w, h), flags=cv2.INTER_NEAREST)
        f.write(rotated_data.tobytes())

from PIL import Image
import os

def rotate_image(file_path, save_dir, degree):
    """
    画像を回転し、元のサイズを保ちながら、はみ出た部分を無視して保存します。
    
    Args:
        file_path (str): 入力画像のファイルパス。
        save_dir (str): 保存先ディレクトリのパス。
        degree (float): 回転角度（時計回り）。
    
    Returns:
        str: 保存された画像のパス。
    """
    try:
        # 入力画像を開く
        with Image.open(file_path) as img:
            original_size = img.size  # 元画像のサイズを取得
            
            # 回転処理を実行（画像全体を回転）
            rotated_img = img.rotate(degree, expand=True)
            
            # 中央から元のサイズでクロップ
            rotated_width, rotated_height = rotated_img.size
            left = (rotated_width - original_size[0]) / 2
            top = (rotated_height - original_size[1]) / 2
            right = left + original_size[0]
            bottom = top + original_size[1]
            cropped_img = rotated_img.crop((left, top, right, bottom))
            
            # 保存先ディレクトリが存在しない場合は作成
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 保存ファイル名の生成
            file_name = os.path.basename(file_path)
            save_path = os.path.join(save_dir, file_name)
            
            # クロップ後の画像を保存
            cropped_img.save(save_path, "PNG")
            
            print(f"画像が保存されました: {save_path}")
            return save_path
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

    
# モデルとデバイスの設定
model_path = "results_2025-05-22/crack_detection_model.pth"
# model_path = "TLresults_2025-05-08/best_transfer_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1,
    activation="sigmoid"
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# 推論処理
raw_image_path = "hyouka/2400_SC.raw"
label_image_path = "hyouka/2400_gt2.png"

raw_image = load_raw_image(raw_image_path, h=168, w=635)
predicted_image = split_and_merge_with_inference(raw_image, model, device, grid_size=32)

# 推論結果の保存
output_path = "results_2025-05-22/predicted_mask.png"
Image.fromarray(predicted_image).save(output_path)
print(f"推論結果を {output_path} に保存しました")

# 精度評価
evaluate_predictions(output_path, label_image_path, "results_2025-05-22/evaluation_results.txt")

