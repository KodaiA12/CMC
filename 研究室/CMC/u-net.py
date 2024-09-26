import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Normalize
from albumentations.pytorch import ToTensorV2

# データセットの作成
class CrackDataset(Dataset):
    def __init__(self, images, masks, augmentation=None):
        self.images = images
        self.masks = masks
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

# データ拡張（Data Augmentation）
def get_augmentation():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2(),
    ])

# U-Net++ モデルの定義（EfficientNet-b4を使用）
def get_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",  # EfficientNet-b4をエンコーダーに使用
        encoder_weights="imagenet",      # ImageNetで事前学習された重みを使用
        in_channels=1,                   # 入力チャンネル数（1: グレースケール画像）
        classes=3                        # クラス数（亀裂のクラス数：0, 1, 亀裂以外）
    )
    return model

# Focal Loss 損失関数の定義
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 確率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# ハイパーパラメータ（表2に基づく）
batch_size = 16  # 表2では16バッチサイズ
learning_rate = 0.001  # 表2に基づく学習率
epochs = 50  # 表2では50エポック

# モデルの準備
model = get_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# オプティマイザの設定（表2に基づき、Adam最適化器を使用）
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# トレーニング関数
def train_model(model, dataloader, optimizer, device, num_epochs=epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = FocalLoss()(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}')

# データセットとデータローダーの作成
# augmentation = get_augmentation()
# dataset = CrackDataset(images, masks, augmentation=augmentation)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルのトレーニング
# train_model(model, dataloader, optimizer, device, num_epochs=epochs)
