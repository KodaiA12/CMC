import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# 事前学習済みモデルのパスを指定
pretrained_model_path = "crack_detection_model.pth"

# モデルの定義 (元のモデルと同じ構造を使用)
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",  # エンコーダーの初期重みはImageNetを使用
    in_channels=1,
    classes=1,
    activation='sigmoid'
)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 事前学習済みモデルのパラメータをロード
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model = model.to(device)

# 転移学習のための設定
# オプション1: エンコーダーを凍結し、デコーダーのみを学習する場合
def freeze_encoder(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # デコーダーのパラメータは学習可能なままにする
    for param in model.decoder.parameters():
        param.requires_grad = True
    
    return model

# オプション2: 全ての層を微調整する場合
def fine_tune_all(model):
    # すべてのパラメータを学習可能にする
    for param in model.parameters():
        param.requires_grad = True
    
    return model

# どちらかの転移学習スタイルを選択
# model = freeze_encoder(model)  # エンコーダを凍結
model = fine_tune_all(model)     # 全ての層を微調整

# データセットの準備 (元のコードを再利用)
class CrackDatasetRaw(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # RAW画像を読み込みと正規化
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        with open(img_path, "rb") as f:
            raw_data = f.read()
            raw_image = np.frombuffer(raw_data, dtype=np.int16).reshape(32, 32)
            raw_image = (raw_image.astype("float32") - raw_image.min()) / (raw_image.max() - raw_image.min())
            raw_image = torch.from_numpy(raw_image).unsqueeze(0)  # [1, H, W]

        # マスク画像の読み込みと正規化
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert("L")  # グレースケールで読み込み
        mask = np.array(mask).astype("float32") / 255.0  # 0~1に正規化
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        return raw_image, mask

# 損失関数
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        eps = 1e-8
        pt = torch.clamp(pt, eps, 1-eps)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

criterion = FocalLoss()

# オプティマイザ - 転移学習では一般的に低い学習率を使用
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# 転移学習用のデータセット (新しいデータセットのパス)
train_dataset = CrackDatasetRaw(
    image_dir="dataset_TL/train/images",  # 新しいRAW画像のディレクトリ
    mask_dir="dataset_TL/train/masks",    # 新しいマスク画像のディレクトリ
)

eval_dataset = CrackDatasetRaw(
    image_dir="dataset_TL/validation/images",
    mask_dir="dataset_TL/validation/masks",
)

# データローダ
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

# 学習設定
num_epochs = 50  # 転移学習では少ないエポック数でも良い結果が得られることが多い

# 保存ディレクトリとファイル名設定
date_str = datetime.now().strftime('%Y-%m-%d')
save_dir = f"TLresults_{date_str}"
os.makedirs(save_dir, exist_ok=True)
csv_file = os.path.join(save_dir, f"transfer_learning_data_{date_str}.csv")

# CSVファイルの準備
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Eval Loss", "Eval Accuracy"])

# 学習履歴を保存するためのリスト
train_losses, eval_losses = [], []
train_accuracies, eval_accuracies = [], []

# 学習ループ
best_eval_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    # トレーニング
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).float()

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
    
    # 最良のモデルを保存
    if eval_accuracy > best_eval_accuracy:
        best_eval_accuracy = eval_accuracy
        best_model_path = os.path.join(save_dir, "best_transfer_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with accuracy: {best_eval_accuracy:.4f}")

# 最終モデルの保存
final_model_path = os.path.join(save_dir, "final_transfer_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"転移学習完了 - 最終モデルを '{final_model_path}' に保存しました")

# グラフのプロットと保存
plt.figure(figsize=(12, 5))

# Lossのプロット
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), eval_losses, label='Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transfer Learning Loss')
plt.legend()

# Accuracyのプロット
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), eval_accuracies, label='Eval Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Transfer Learning Accuracy')
plt.legend()

# グラフの保存
graph_path = os.path.join(save_dir, f"transfer_learning_graph_{date_str}.png")
plt.tight_layout()
plt.savefig(graph_path)
print(f"グラフを '{graph_path}' に保存しました")