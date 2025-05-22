# GAN-based Crack Detection using U-Net Generator and PatchGAN Discriminator (32x32 cropped) 

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import glob
import cv2

# Dataset class for cropped 32x32 raw images and masks
class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.raw")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        h, w = 32, 32
        with open(self.image_paths[idx], "rb") as f:
            image = np.frombuffer(f.read(), dtype=np.int16).reshape(h, w)
        image = (image.astype(np.float32) - image.min()) / (image.max() - image.min())
        image = torch.from_numpy(image).unsqueeze(0)

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize((w, h), Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# U-Net like Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training function

def train_gan(image_dir, mask_dir, save_dir, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CrackDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    g_optim = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)

            # Generator
            fake_mask = generator(img)
            pred_fake = discriminator(torch.cat([img, fake_mask], 1))
            valid = torch.ones_like(pred_fake)
            fake = torch.zeros_like(pred_fake)

            g_loss = bce(pred_fake, valid) + 100 * l1(fake_mask, mask)
            g_optim.zero_grad(); g_loss.backward(); g_optim.step()

            # Discriminator
            real_pred = discriminator(torch.cat([img, mask], 1))
            fake_pred = discriminator(torch.cat([img, fake_mask.detach()], 1))
            real_loss = bce(real_pred, valid)
            fake_loss = bce(fake_pred, fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_optim.zero_grad(); d_loss.backward(); d_optim.step()

        print(f"Epoch {epoch+1}/{num_epochs}  G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")

    torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, "discriminator.pth"))

# Inference and Evaluation script

def split_and_merge_with_gan_inference(image, generator, device, grid_size=32):
    """
    大きな画像を分割してGANのGeneratorで推論し、再結合する関数。
    
    Args:
        image: 入力画像 (numpy array, shape: [H, W])
        generator: 学習済みのUNetGenerator
        device: torch.device
        grid_size: パッチサイズ (default: 32)
    
    Returns:
        reconstructed_image: 再構成されたマスク画像 (numpy array, shape: [H, W])
    """
    generator.eval()  # 推論モードに設定
    h, w = image.shape

    # 画像をパディング（grid_sizeの倍数にする）
    y_surp, x_surp = h % grid_size, w % grid_size
    pad_h = grid_size - y_surp if y_surp != 0 else 0
    pad_w = grid_size - x_surp if x_surp != 0 else 0
    
    padded_image = np.pad(
        image, ((0, pad_h), (0, pad_w)), "constant", constant_values=0
    )

    predicted_patches = []
    ph, pw = padded_image.shape

    # パッチごとに推論実行
    for i in range(0, ph, grid_size):
        for j in range(0, pw, grid_size):
            # パッチを切り出し
            patch = padded_image[i : i + grid_size, j : j + grid_size]
            
            # 正規化（0-1範囲）
            if patch.max() > patch.min():
                normalized_patch = (patch.astype('float32') - patch.min()) / (patch.max() - patch.min())
            else:
                normalized_patch = patch.astype('float32')
            
            # Tensorに変換 [1, 1, H, W]
            input_patch = torch.from_numpy(normalized_patch).unsqueeze(0).unsqueeze(0).to(device)
            
            # Generatorで推論（Discriminatorは使用しない）
            with torch.no_grad():
                fake_mask = generator(input_patch)  # Generatorのみ使用
                prediction = fake_mask.squeeze().cpu().numpy()
            
            # 二値化（閾値0.5）
            predicted_patch = (prediction > 0.5).astype(np.uint8) * 255
            predicted_patches.append(predicted_patch)

    # パッチを再結合
    reconstructed_image = np.zeros((ph, pw), dtype=np.uint8)
    idx = 0
    for i in range(0, ph, grid_size):
        for j in range(0, pw, grid_size):
            reconstructed_image[i : i + grid_size, j : j + grid_size] = predicted_patches[idx]
            idx += 1
    
    # 元のサイズに戻す（パディング除去）
    reconstructed_image = reconstructed_image[:h, :w]
    return reconstructed_image


def infer_and_evaluate_with_patches(image_path, gt_path, generator_path, output_path, grid_size=32, result_txt=None):
    """
    GANのGeneratorを使用してパッチベース推論と評価を行う関数
    
    Args:
        image_path: 入力画像パス (.raw)
        gt_path: 正解マスクパス (.png)
        generator_path: 学習済みGeneratorのパス (.pth)
        output_path: 出力マスク保存パス (.png)
        grid_size: パッチサイズ
        result_txt: 結果保存用txtファイルパス (Noneの場合は自動生成)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = 168, 635  # 画像サイズ

    # 入力画像読み込み
    with open(image_path, "rb") as f:
        img = np.frombuffer(f.read(), dtype=np.int16).reshape(h, w)
    
    # Generatorモデル読み込み
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    
    # パッチベース推論実行
    pred_mask = split_and_merge_with_gan_inference(img, generator, device, grid_size)
    
    # 結果保存
    Image.fromarray(pred_mask).save(output_path)
    
    # 正解データ読み込み
    gt = np.array(Image.open(gt_path).convert("L"))
    
    # 評価指標計算
    TP = np.sum((pred_mask == 255) & (gt == 255))
    TN = np.sum((pred_mask == 0) & (gt == 0))
    FP = np.sum((pred_mask == 255) & (gt == 0))
    FN = np.sum((pred_mask == 0) & (gt == 255))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # IoU追加
    intersection = TP
    union = TP + FP + FN
    iou = intersection / (union + 1e-8)

    # 結果を文字列として整理
    results = (f"推論結果画像: {output_path}\n"
               f"正解ラベル画像: {gt_path}\n"
               f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n"
               f"精度: {accuracy:.4f}, 再現率: {recall:.4f}, 適合率: {precision:.4f}, F値: {f1:.4f}\n")
    
    # 結果をコンソールに表示
    print(results)
    
    # txtファイルに保存
    if result_txt is None:
        # 自動でtxtファイル名を生成
        base_name = os.path.splitext(output_path)[0]
        result_txt = f"{base_name}_results.txt"
    
    with open(result_txt, 'w', encoding='utf-8') as f:
        f.write(results)
    
    print(f"Results saved to: {result_txt}")
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'iou': iou,
        'confusion_matrix': {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    }

# 使用例
train_gan("dataset_GAN/images", "dataset_GAN/masks", "gan_result")

infer_and_evaluate_with_patches(
    "hyouka/2400_SC.raw", 
    "hyouka/2400_gt2.png", 
    "gan_result/generator.pth", 
    "gan_result/predicted_masks.png",
    grid_size=32,
    result_txt="evaluation_results.txt"  # 指定しない場合は自動生成
    )