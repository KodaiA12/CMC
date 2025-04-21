import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def generate_synthetic_crack_image(
    width=574,
    height=130,
    num_images=10,
    save_path="./synthetic_dataset",
    crack_thickness_range=(1, 4),
    crack_length_range=(20, 100),
    material_intensity_range=(250, 320),
    void_intensity_range=(115, 170),  # 素材がない部分の明度
    crack_intensity_range=(350, 600),
    noise_stddev=5,
    min_cracks=10,
    max_cracks=30,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    
    images_dir = Path(save_path) / "images"
    masks_dir = Path(save_path) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(num_images), desc="Generating images"):
        # 試験片形状の背景を作成
        material_mask, image = create_test_specimen_background(
            width, height, 
            material_intensity_range,
            void_intensity_range
        )
        
        # ノイズを追加 (素材部分のみ)
        noise = np.random.normal(0, noise_stddev, (height, width))
        # 素材部分にのみノイズを適用
        image = image + (noise * material_mask)
        image = np.clip(image, 0, 65535).astype(np.uint16)
        
        # クラックマスクを作成
        crack_mask = np.zeros((height, width), dtype=np.uint8)
        
        num_cracks = np.random.randint(min_cracks, max_cracks + 1)
        
        # クラックを描画する際に、素材マスクを確実に参照
        for _ in range(num_cracks):
            # 素材の中に縦線を描画
            crack_thickness = np.random.randint(*crack_thickness_range)
            crack_intensity = sample_decreasing_probability(crack_intensity_range[0], crack_intensity_range[1])
            
            # 線を引く位置をランダムに選択
            x = np.random.randint(0, width)
            
            # その位置での素材の上端と下端を見つける
            material_column = material_mask[:, x]
            if np.any(material_column > 0.5):  # しきい値を追加して明確に素材部分を特定
                # 素材がある場所を特定
                material_indices = np.where(material_column > 0.5)[0]
                if len(material_indices) > 0:
                    y_min = material_indices.min()
                    y_max = material_indices.max()
                    
                    # 境界から少し内側に余裕を持たせる（重要な修正点）
                    margin = max(2, crack_thickness)  # クラックの厚さに応じた余裕
                    y_min += margin
                    y_max -= margin
                    
                    # 素材の高さが最小クラック長より大きい場合のみ描画
                    if y_max - y_min + 1 > crack_length_range[0]:
                        # クラックの長さが素材の高さを超えないようにする
                        max_possible_length = y_max - y_min + 1
                        low = min(crack_length_range[0], max_possible_length - 1)
                        high = min(crack_length_range[1], max_possible_length)
                        
                        if low < high:
                            crack_length = np.random.randint(low, high)
                        else:
                            crack_length = low
                        
                        # クラックの開始位置（素材内に収まるように）
                        y_start = np.random.randint(y_min, y_max - crack_length + 1)
                        y_end = y_start + crack_length
                        
                        # クラックを描画（素材内のみ）
                        cv2.line(image, (x, y_start), (x, y_end), color=int(crack_intensity), thickness=crack_thickness)
                        cv2.line(crack_mask, (x, y_start), (x, y_end), color=255, thickness=crack_thickness)
        
        # RAWファイルとして保存
        image_path = images_dir / f"image_{i:04d}.raw"
        image.astype(np.uint16).tofile(image_path)
        
        # マスクを保存
        mask_path = masks_dir / f"mask_{i:04d}.png"
        cv2.imwrite(str(mask_path), crack_mask)
    
    return f"Saved {num_images} synthetic images with masks to {save_path}"

def sample_decreasing_probability(min_val, max_val):
    """
    高い値ほど出現確率が低くなるようにサンプリングする関数
    """
    range_size = max_val - min_val
    x = np.random.random()
    value = min_val + range_size * (1 - x**2)
    return int(value)

def create_test_specimen_background(width, height, material_intensity_range, void_intensity_range):
    """
    試験片形状（中央がくびれた形）の背景を作成する関数
    """
    # まず素材がない部分の画像を作成
    image = np.random.randint(
        void_intensity_range[0], 
        void_intensity_range[1], 
        size=(height, width),
        dtype=np.uint16
    )
    
    # 素材部分のマスクを作成（中央がくびれた形）
    material_mask = np.zeros((height, width), dtype=np.float32)
    
    # パラメータ設定
    mid_width_ratio = 0.8  # 中央部の幅の比率
    edge_width_ratio = 1.0  # 端部の幅の比率
    
    # 各x座標での試験片の幅を計算
    for x in range(width):
        # x座標の位置を0～1で正規化
        rel_x = x / width
        
        # 位置に応じた幅の比率を計算（中央が細く、端が太い）
        # 2次関数で滑らかに変化させる
        width_ratio = mid_width_ratio + (edge_width_ratio - mid_width_ratio) * (2 * (rel_x - 0.5))**2
        
        # この位置での試験片の半幅を計算
        half_specimen_width = int((height * width_ratio) / 2)
        
        # 中心位置を計算
        center_y = height // 2
        
        # 上下のy座標の範囲を計算
        y_top = center_y - half_specimen_width
        y_bottom = center_y + half_specimen_width
        
        # マスクに値を設定
        material_mask[y_top:y_bottom, x] = 1.0
    
    # 端のエッジを少しランダムにして自然な形状にする
    for x in range(width):
        col = material_mask[:, x]
        if np.any(col):
            y_top = np.where(col > 0)[0][0]
            y_bottom = np.where(col > 0)[0][-1]
            
            # 上下の端に小さなランダム変動を加える
            rand_top = np.random.randint(-2, 3) if x > 0 and x < width-1 else 0
            rand_bottom = np.random.randint(-2, 3) if x > 0 and x < width-1 else 0
            
            # 更新されたy位置を計算
            new_y_top = max(0, min(y_top + rand_top, height - 1))
            new_y_bottom = max(0, min(y_bottom + rand_bottom, height - 1))
            
            # マスクを更新
            material_mask[:, x] = 0
            material_mask[new_y_top:new_y_bottom+1, x] = 1.0
    
    # エッジをぼかす
    material_mask = cv2.GaussianBlur(material_mask, (5, 5), 0)
    
    # 素材部分の明度を設定
    material = np.random.randint(
        material_intensity_range[0], 
        material_intensity_range[1], 
        size=(height, width),
        dtype=np.uint16
    )
    
    # 素材部分と素材がない部分を組み合わせる
    image = image * (1 - material_mask) + material * material_mask
    
    return material_mask, image.astype(np.uint16)

# サンプル実行
if __name__ == "__main__":
    generate_synthetic_crack_image(
        num_images=5, 
        save_path="./synthetic_crack_dataset3",
        material_intensity_range=(250, 320),
        void_intensity_range=(115, 170),
        crack_intensity_range=(350, 600),
        crack_thickness_range=(1, 5)
    )