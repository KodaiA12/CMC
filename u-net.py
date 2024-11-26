import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
import numpy as np
from PIL import Image
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os
import cv2

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
class CrackDatasetRaw(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        RAW画像とマスク画像を扱うデータセットクラス。
        
        Parameters:
        - image_dir (str): RAW画像ディレクトリ
        - mask_dir (str): マスク画像ディレクトリ
        - transform (callable, optional): 画像とマスクへの変換
        """
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



# モデルの作成
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
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
        print("BCE:",BCE_loss)
        pt = torch.exp(-BCE_loss)
        eps = 1e-8
        pt = torch.clamp(pt, eps, 1-eps)
        print("pt:", pt)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        print("F_loss:",F_loss)
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

criterion = FocalLoss()

# オプティマイザの設定
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# トレーニングデータセット
train_dataset = CrackDatasetRaw(
    image_dir="dataset/train/images",  # RAW画像のディレクトリ
    mask_dir="dataset/train/masks",    # マスク画像のディレクトリ
)

# 検証データセット
eval_dataset = CrackDatasetRaw(
    image_dir="dataset/validation/images",
    mask_dir="dataset/validation/masks",
)

# データローダ
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=False)

# 学習設定
num_epochs = 1

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
        # print(f"Image Min: {images.min()}, Max: {images.max()}")
        # print(f"Mask Min: {masks.min()}, Max: {masks.max()}")
        images = images.to(device)
        masks = masks.to(device).float()  # 明示的にfloatに変換

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


# 学習済みモデルをロード
model.load_state_dict(torch.load(model_path, map_location=device))

# 推論対象のRAW画像をロード
test_image = load_raw_image("hyouka/2400_1_SC.raw", 100, 460)  # 460x100の画像をロード

# 推論と再結合
predicted_image = split_and_merge_with_inference(test_image, model, device, grid_size=32)

# 推論結果を保存
output_path = "predicted_mask.png"
result_image = Image.fromarray(predicted_image)
result_image.save(output_path)
print(f"推論結果を{output_path}に保存しました")
