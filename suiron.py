import torch
import numpy as np
from PIL import Image
import os


def load_raw_image(file_path, h, w):
    """
    RAW画像をnumpy配列として読み込む関数。
    
    Parameters:
    - file_path (str): RAWファイルのパス
    - h (int): 画像の高さ
    - w (int): 画像の幅
    
    Returns:
    - data (numpy array): 画像データ
    """
    with open(file_path, "rb") as f:
        raw_data = f.read()
        image = np.frombuffer(raw_data, dtype=np.int16).reshape(h, w)
    return image


def split_and_merge_with_inference(image, model, device, grid_size=32):
    """
    大きな画像を指定されたグリッドサイズで分割し、モデルで推論し、再結合する関数。

    Parameters:
    - image (numpy array): 推論対象のRAW画像データ
    - model (torch.nn.Module): 学習済みモデル
    - device (torch.device): 使用するデバイス（CPUまたはGPU）
    - grid_size (int): 分割するグリッドのサイズ (デフォルト: 32)

    Returns:
    - reconstructed_image (numpy array): モデルの推論結果を再結合した画像
    """
    model.eval()  # 推論モードに設定

    # 元画像のサイズ
    h, w = image.shape

    # 余りを計算し、画像をパディング
    y_surp, x_surp = h % grid_size, w % grid_size
    padded_image = np.pad(
        image, ((0, grid_size - y_surp), (0, grid_size - x_surp)), "constant"
    )

    # パッチごとに推論し、結果を保存するリスト
    predicted_patches = []

    # パッチの切り取りと推論
    ph, pw = padded_image.shape
    for i in range(0, ph, grid_size):
        for j in range(0, pw, grid_size):
            # パッチの切り取り
            patch = padded_image[i : i + grid_size, j : j + grid_size]

            # パッチの前処理
            normalized_patch = (patch.astype('float32') - patch.min()) / (patch.max() - patch.min())
            input_patch = torch.from_numpy(normalized_patch).unsqueeze(0).unsqueeze(0).to(device)  # [B, C, H, W]

            # モデルで推論
            with torch.no_grad():
                output = model(input_patch)
                prediction = output.squeeze().cpu().numpy()

            # 推論結果を二値化して保存
            predicted_patch = (prediction > 0.5).astype(np.uint8) * 255
            predicted_patches.append(predicted_patch)

    # 推論結果を再結合
    reconstructed_image = np.zeros_like(padded_image, dtype=np.uint8)
    idx = 0
    for i in range(0, ph, grid_size):
        for j in range(0, pw, grid_size):
            reconstructed_image[i : i + grid_size, j : j + grid_size] = predicted_patches[idx]
            idx += 1

    # パディングを取り除いて元のサイズに戻す
    reconstructed_image = reconstructed_image[:h, :w]
    return reconstructed_image


def calculate_crack_accuracy(predicted_image, label_image):
    """
    推論結果とラベルデータの白い部分の正解率を計算する関数。

    Parameters:
    - predicted_image (numpy array): 推論結果（0または255の二値画像）
    - label_image (numpy array): ラベルデータ（0または255の二値画像）

    Returns:
    - accuracy (float): 白い部分（亀裂部分）の正解率
    """
    # 二値化（0または1に変換）
    pred_binary = (predicted_image > 128).astype(np.uint8)
    label_binary = (label_image > 128).astype(np.uint8)

    # 白い部分のみの比較
    crack_area = label_binary.sum()
    if crack_area == 0:
        raise ValueError("ラベル画像に白い部分が含まれていません。")

    correct_predictions = (pred_binary & label_binary).sum()
    accuracy = correct_predictions / crack_area
    return accuracy


# モデルとデバイスの設定
model_path = "results_2024-11-29/crack_detection_model.pth"  # 学習済みモデルのパス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの定義（Unet++を使用している例）
import segmentation_models_pytorch as smp
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1,
    activation="sigmoid"
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# 推論対象のRAW画像とラベル画像の読み込み
raw_image_path = "hyouka/2400_SC.raw"  # 推論対象のRAW画像
label_image_path = "2400.png"  # ラベル画像（PNG形式）

raw_image = load_raw_image(raw_image_path, h=100, w=460)  # RAW画像サイズを指定
label_image = np.array(Image.open(label_image_path).convert("L"))  # グレースケール画像

# 推論と正解率の計算
predicted_image = split_and_merge_with_inference(raw_image, model, device, grid_size=32)
accuracy = calculate_crack_accuracy(predicted_image, label_image)

# 推論結果を保存
output_path = "predicted_mask.png"
result_image = Image.fromarray(predicted_image)
result_image.save(output_path)
print(f"推論結果を {output_path} に保存しました")

# 正解率の表示
print(f"白い部分（亀裂部分）の正解率: {accuracy:.2%}")
