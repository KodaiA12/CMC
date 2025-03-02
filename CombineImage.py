import numpy as np
from PIL import Image
"~~~評価用~~~"

def create_difference_image(predicted_image, label_image):
    """
    推論結果とラベルデータの差分画像を作成する関数。

    Parameters:
    - predicted_image (numpy array): 推論結果（0または255の二値画像）
    - label_image (numpy array): ラベルデータ（0または255の二値画像）

    Returns:
    - diff_image (PIL Image): 差分を色分けした画像
    """
    # 二値化
    pred_binary = (predicted_image > 128).astype(np.uint8)
    label_binary = (label_image > 128).astype(np.uint8)

    # 一致する部分（正解部分）: 白色
    true_positive = (pred_binary & label_binary) * 255

    # 推論が過剰な部分（余分に亀裂と判断）: 赤色
    false_positive = (pred_binary & ~label_binary) * 255

    # 推論が不足している部分（検知できなかった亀裂）: 緑色
    false_negative = (~pred_binary & label_binary) * 255

    # 差分画像を作成
    diff_image = np.zeros((pred_binary.shape[0], pred_binary.shape[1], 3), dtype=np.uint8)
    diff_image[..., 0] = false_positive  # 赤チャンネル (FP)
    diff_image[..., 1] = false_negative  # 緑チャンネル (FN)
    diff_image[..., 0:3] = diff_image[..., 0:3] + np.stack([true_positive]*3, axis=2) // 3  # 白チャンネル (TP)

    return Image.fromarray(diff_image)


# 正解画像と推論画像を読み込む
label_image_path = "hyouka/2400_gt.png"       # ラベル画像のパス
predicted_image_path = "predicted_mask.png"        # 推論結果画像のパス

label_image = np.array(Image.open(label_image_path).convert("L"))       # ラベル画像をグレースケールで読み込み
predicted_image = np.array(Image.open(predicted_image_path).convert("L"))  # 推論結果画像をグレースケールで読み込み

# 差分画像を生成
diff_image = create_difference_image(predicted_image, label_image)

# 差分画像を保存
output_path = "difference_image_with_tp.png"
diff_image.save(output_path)
print(f"差分画像を {output_path} に保存しました。")
