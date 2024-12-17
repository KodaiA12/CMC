import torch
import numpy as np
from PIL import Image
import os
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


def rotate_image(image, degree):
    """
    画像を指定した角度で回転させる関数。

    Parameters:
    - image (numpy array): 入力画像
    - degree (float): 回転角度（度単位, 正は反時計回り）

    Returns:
    - rotated_image (numpy array): 回転後の画像
    """
    # 画像を回転（回転後、元のサイズにリサイズ）
    rotated_image = rotate(image, degree, reshape=False, mode="constant", cval=0)
    return rotated_image


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


def evaluate_predictions(result_file_path, gt_file_path, save_path):
    """
    推論結果とラベルデータの比較（TP, TN, FP, FN）から精度を計算し、結果を保存する関数。
    """
    img = cv2.imread(result_file_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_file_path, cv2.IMREAD_GRAYSCALE)

    if img is None or gt is None:
        raise FileNotFoundError("推論画像または正解画像が見つかりません。パスを確認してください。")

    TP, FP, TN, FN = 0, 0, 0, 0
    Y, X = gt.shape

    for y in range(Y):
        for x in range(X):
            if img[y, x] == 255 and gt[y, x] == 255:
                TP += 1
            elif img[y, x] == 0 and gt[y, x] == 0:
                TN += 1
            elif img[y, x] == 0 and gt[y, x] == 255:
                FN += 1
            elif img[y, x] == 255 and gt[y, x] == 0:
                FP += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f_measure = (2 * recall * precision) / (recall + precision) if (recall + precision) != 0 else 0

    result_text = (
        f"推論結果画像: {result_file_path}\n"
        f"正解ラベル画像: {gt_file_path}\n"
        f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
        f"精度: {accuracy:.4f}, 再現率: {recall:.4f}, 適合率: {precision:.4f}, F値: {f_measure:.4f}\n"
    )
    print(result_text)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(result_text)
    print(f"評価結果を {save_path} に保存しました。")


# モデルとデバイスの設定
model_path = "results_2024-12-10/crack_detection_model.pth"
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
degree = 15  # 画像を回転させる角度を指定
raw_image_path = "hyouka/2400_1_SC.raw"
label_image_path = "hyouka/2400_gt.png"

# 画像の読み込みと回転
raw_image = load_raw_image(raw_image_path, h=175, w=638)
rotated_raw_image = rotate_image(raw_image, degree)

label_image = np.array(Image.open(label_image_path).convert("L"))
rotated_label_image = rotate_image(label_image, degree)

# 傾いた画像で推論
predicted_image = split_and_merge_with_inference(rotated_raw_image, model, device, grid_size=32)

# 推論結果の保存
output_path = "predicted_mask_rotated.png"
Image.fromarray(predicted_image).save(output_path)
print(f"推論結果を {output_path} に保存しました")

# 精度評価（傾いた正解画像と比較）
rotated_label_path = "rotated_label.png"
Image.fromarray(rotated_label_image).save(rotated_label_path)
evaluate_predictions(output_path, rotated_label_path, "results_2024-12-10/evaluation_results.txt")
