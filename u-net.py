import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from PIL import Image
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os
import cv2

# モデルの作成
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation='sigmoid'
)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 損失関数としてFocal Lossを使用
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

criterion = FocalLoss()

# オプティマイザの設定
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# データローダの設定
train_dataset = "dataset/train"  # トレーニングデータセットを定義
eval_dataset = "dataset/validation"  # 評価データセットを定義
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=False)

# 学習設定
num_epochs = 500

# 保存ディレクトリとファイル名設定
date_str = datetime.now().strftime('%Y-%m-%d')
save_dir = f"results_{date_str}"
os.makedirs(save_dir, exist_ok=True)
csv_file = os.path.join(save_dir, f"traindata_{date_str}.csv")

# CSVファイルの準備
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Eval Loss", "Eval Accuracy"])

# 学習履歴を保存するためのリスト
train_losses, eval_losses = [], []
train_accuracies, eval_accuracies = [], []

# 学習ループ
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # トレーニング
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == masks).sum().item()
        total += masks.numel()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 評価モード
    model.eval()
    eval_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, masks in eval_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            eval_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == masks).sum().item()
            total += masks.numel()

    eval_loss /= len(eval_loader)
    eval_accuracy = correct / total
    eval_losses.append(eval_loss)
    eval_accuracies.append(eval_accuracy)

    # CSVに出力
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, train_loss, train_accuracy, eval_loss, eval_accuracy])

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_accuracy:.4f}")

# 学習済みモデルの保存
model_path = os.path.join(save_dir, "crack_detection_model.pth")
torch.save(model.state_dict(), model_path)
print(f"学習完了 - モデルを '{model_path}' に保存しました")

# グラフのプロットと保存
plt.figure(figsize=(12, 5))

# Lossのプロット
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), eval_losses, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()

# Accuracyのプロット
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), eval_accuracies, label='Eval Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Evaluation Accuracy')
plt.legend()

# グラフの保存
graph_path = os.path.join(save_dir, f"training_graph_{date_str}.png")
plt.tight_layout()
plt.savefig(graph_path)
plt.show()

print(f"グラフを '{graph_path}' に保存しました")

def load_raw_image(file_path, h, w):
    """
    .raw画像をnumpy配列として読み込む関数。
    
    Parameters:
    - file_path (str): rawファイルのパス
    - h (int): 画像の高さ
    - w (int): 画像の幅
    
    Returns:
    - data (numpy array): 画像データ
    """
    with open(file_path, "rb") as f:
        raw_data = f.read()
        image = np.frombuffer(raw_data, dtype=np.int16).reshape(h, w)
    return image

def split_into_patches(image, patch_size=(50, 50)):
    """
    画像を指定されたサイズのパッチに分割する関数。

    Parameters:
    - image (numpy array): 元画像
    - patch_size (tuple): 分割するパッチのサイズ (高さ, 幅)

    Returns:
    - patches (list): パッチのリスト
    - positions (list): パッチの開始位置 (x, y) のリスト
    """
    h, w = image.shape
    ph, pw = patch_size
    patches, positions = [], []

    for y in range(0, h, ph):
        for x in range(0, w, pw):
            patch = image[y:y + ph, x:x + pw]
            patches.append(patch)
            positions.append((x, y))
    return patches, positions

def merge_patches(patches, positions, output_shape):
    """
    分割されたパッチを元の画像サイズに結合する関数。

    Parameters:
    - patches (list): パッチのリスト
    - positions (list): パッチの開始位置 (x, y) のリスト
    - output_shape (tuple): 元の画像サイズ (高さ, 幅)

    Returns:
    - merged_image (numpy array): 再構成された画像
    """
    h, w = output_shape
    merged_image = np.zeros((h, w), dtype=np.uint8)
    for patch, (x, y) in zip(patches, positions):
        ph, pw = patch.shape
        merged_image[y:y + ph, x:x + pw] = patch
    return merged_image

def predict_large_image(image_path, model, device, output_path="predicted_mask.png", original_size=(460, 100), patch_size=(50, 50)):
    """
    大きな画像をパッチごとに予測し、結果を再構成する関数。

    Parameters:
    - image_path (str): 予測対象のraw画像ファイルパス
    - model (torch.nn.Module): 学習済みモデル
    - device (torch.device): 使用するデバイス（CPUまたはGPU）
    - output_path (str): 予測結果の保存先
    - original_size (tuple): 入力画像の元サイズ (高さ, 幅)
    - patch_size (tuple): モデルに入力するパッチのサイズ (高さ, 幅)
    """
    model.eval()

    # raw画像を読み込み
    h, w = original_size
    raw_image = load_raw_image(image_path, h, w)

    # 画像をパッチに分割
    patches, positions = split_into_patches(raw_image, patch_size)

    # 各パッチを予測
    predicted_patches = []
    for patch in patches:
        # パッチをリサイズして正規化
        resized_patch = cv2.resize(patch, (50, 50), interpolation=cv2.INTER_LINEAR)
        normalized_patch = (resized_patch.astype('float32') - resized_patch.min()) / (resized_patch.max() - resized_patch.min())
        input_patch = torch.from_numpy(normalized_patch).unsqueeze(0).unsqueeze(0).to(device)  # [B, C, H, W]

        # 推論
        with torch.no_grad():
            output = model(input_patch)
            prediction = output.squeeze().cpu().numpy()
            predicted_patches.append((prediction > 0.5).astype(np.uint8) * 255)

    # パッチを結合して元のサイズに戻す
    predicted_mask = merge_patches(predicted_patches, positions, output_shape=(h, w))

    # PNG形式で保存
    result_image = Image.fromarray(predicted_mask)
    result_image.save(output_path)
    print(f"予測結果を{output_path}に保存しました")

# 学習済みモデルを読み込み
model.load_state_dict(torch.load("crack_detection_model.pth", map_location=device))

# 推論の実行例
test_image_path = "./hyouka/2400_1_SC.raw"  # 予測対象のraw画像パス
predict_large_image(test_image_path, model, device, output_path="predicted_mask.png", original_size=(460, 100), patch_size=(50, 50))