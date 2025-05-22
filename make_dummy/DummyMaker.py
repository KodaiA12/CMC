# import os
# import numpy as np
# import cv2
# from pathlib import Path
# from tqdm import tqdm

# def generate_synthetic_crack_image(
#     width=574,
#     height=130,
#     num_images=10,
#     save_path="./synthetic_dataset",
#     crack_thickness_range=(1, 4),
#     crack_length_range=(20, 100),
#     material_intensity_range=(250, 320),
#     void_intensity_range=(115, 170),  # 素材がない部分の明度
#     crack_intensity_range=(350, 600),
#     noise_stddev=5,
#     min_cracks=10,
#     max_cracks=30,
#     seed=None,
# ):
#     if seed is not None:
#         np.random.seed(seed)
    
#     images_dir = Path(save_path) / "images"
#     masks_dir = Path(save_path) / "masks"
#     images_dir.mkdir(parents=True, exist_ok=True)
#     masks_dir.mkdir(parents=True, exist_ok=True)
    
#     for i in tqdm(range(num_images), desc="Generating images"):
#         # 試験片形状の背景を作成
#         material_mask, image = create_test_specimen_background(
#             width, height, 
#             material_intensity_range,
#             void_intensity_range
#         )
        
#         # ノイズを追加 (素材部分のみ)
#         noise = np.random.normal(0, noise_stddev, (height, width))
#         # 素材部分にのみノイズを適用
#         image = image + (noise * material_mask)
#         image = np.clip(image, 0, 65535).astype(np.uint16)
        
#         # クラックマスクを作成
#         crack_mask = np.zeros((height, width), dtype=np.uint8)
        
#         num_cracks = np.random.randint(min_cracks, max_cracks + 1)
        
#         # クラックを描画する際に、素材マスクを確実に参照
#         for _ in range(num_cracks):
#             # 素材の中に縦線を描画
#             crack_thickness = np.random.randint(*crack_thickness_range)
#             crack_intensity = sample_decreasing_probability(crack_intensity_range[0], crack_intensity_range[1])
            
#             # 線を引く位置をランダムに選択
#             x = np.random.randint(0, width)
            
#             # その位置での素材の上端と下端を見つける
#             material_column = material_mask[:, x]
#             if np.any(material_column > 0.5):  # しきい値を追加して明確に素材部分を特定
#                 # 素材がある場所を特定
#                 material_indices = np.where(material_column > 0.5)[0]
#                 if len(material_indices) > 0:
#                     y_min = material_indices.min()
#                     y_max = material_indices.max()
                    
#                     # 境界から少し内側に余裕を持たせる（重要な修正点）
#                     margin = max(2, crack_thickness)  # クラックの厚さに応じた余裕
#                     y_min += margin
#                     y_max -= margin
                    
#                     # 素材の高さが最小クラック長より大きい場合のみ描画
#                     if y_max - y_min + 1 > crack_length_range[0]:
#                         # クラックの長さが素材の高さを超えないようにする
#                         max_possible_length = y_max - y_min + 1
#                         low = min(crack_length_range[0], max_possible_length - 1)
#                         high = min(crack_length_range[1], max_possible_length)
                        
#                         if low < high:
#                             crack_length = np.random.randint(low, high)
#                         else:
#                             crack_length = low
                        
#                         # クラックの開始位置（素材内に収まるように）
#                         y_start = np.random.randint(y_min, y_max - crack_length + 1)
#                         y_end = y_start + crack_length
                        
#                         # クラックを描画（素材内のみ）
#                         cv2.line(image, (x, y_start), (x, y_end), color=int(crack_intensity), thickness=crack_thickness)
#                         cv2.line(crack_mask, (x, y_start), (x, y_end), color=255, thickness=crack_thickness)
        
#         # RAWファイルとして保存
#         image_path = images_dir / f"d{i:04d}_SC.raw"
#         image.astype(np.uint16).tofile(image_path)
        
#         # マスクを保存
#         mask_path = masks_dir / f"d{i:04d}_gt.png"
#         cv2.imwrite(str(mask_path), crack_mask)
    
#     return f"Saved {num_images} synthetic images with masks to {save_path}"

# def sample_decreasing_probability(min_val, max_val):
#     """
#     高い値ほど出現確率が低くなるようにサンプリングする関数
#     """
#     range_size = max_val - min_val
#     x = np.random.random()
#     value = min_val + range_size * (1 - x**2)
#     return int(value)

# def create_test_specimen_background(width, height, material_intensity_range, void_intensity_range):
#     """
#     試験片形状（中央がくびれた形）の背景を作成する関数
#     """
#     # まず素材がない部分の画像を作成
#     image = np.random.randint(
#         void_intensity_range[0], 
#         void_intensity_range[1], 
#         size=(height, width),
#         dtype=np.uint16
#     )
    
#     # 素材部分のマスクを作成（中央がくびれた形）
#     material_mask = np.zeros((height, width), dtype=np.float32)
    
#     # パラメータ設定
#     mid_width_ratio = 0.8  # 中央部の幅の比率
#     edge_width_ratio = 1.0  # 端部の幅の比率
    
#     # 各x座標での試験片の幅を計算
#     for x in range(width):
#         # x座標の位置を0～1で正規化
#         rel_x = x / width
        
#         # 位置に応じた幅の比率を計算（中央が細く、端が太い）
#         # 2次関数で滑らかに変化させる
#         width_ratio = mid_width_ratio + (edge_width_ratio - mid_width_ratio) * (2 * (rel_x - 0.5))**2
        
#         # この位置での試験片の半幅を計算
#         half_specimen_width = int((height * width_ratio) / 2)
        
#         # 中心位置を計算
#         center_y = height // 2
        
#         # 上下のy座標の範囲を計算
#         y_top = center_y - half_specimen_width
#         y_bottom = center_y + half_specimen_width
        
#         # マスクに値を設定
#         material_mask[y_top:y_bottom, x] = 1.0
    
#     # 端のエッジを少しランダムにして自然な形状にする
#     for x in range(width):
#         col = material_mask[:, x]
#         if np.any(col):
#             y_top = np.where(col > 0)[0][0]
#             y_bottom = np.where(col > 0)[0][-1]
            
#             # 上下の端に小さなランダム変動を加える
#             rand_top = np.random.randint(-2, 3) if x > 0 and x < width-1 else 0
#             rand_bottom = np.random.randint(-2, 3) if x > 0 and x < width-1 else 0
            
#             # 更新されたy位置を計算
#             new_y_top = max(0, min(y_top + rand_top, height - 1))
#             new_y_bottom = max(0, min(y_bottom + rand_bottom, height - 1))
            
#             # マスクを更新
#             material_mask[:, x] = 0
#             material_mask[new_y_top:new_y_bottom+1, x] = 1.0
    
#     # エッジをぼかす
#     material_mask = cv2.GaussianBlur(material_mask, (5, 5), 0)
    
#     # 素材部分の明度を設定
#     material = np.random.randint(
#         material_intensity_range[0], 
#         material_intensity_range[1], 
#         size=(height, width),
#         dtype=np.uint16
#     )
    
#     # 素材部分と素材がない部分を組み合わせる
#     image = image * (1 - material_mask) + material * material_mask
    
#     return material_mask, image.astype(np.uint16)

# # サンプル実行
# if __name__ == "__main__":
#     generate_synthetic_crack_image(
#         num_images=5000, 
#         save_path="./synthetic_crack_dataset_poli",
#         material_intensity_range=(250, 320),
#         void_intensity_range=(115, 170),
#         crack_intensity_range=(350, 600),
#         crack_thickness_range=(1, 5)
#     )

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
    material_intensity_range=(250, 320),
    void_intensity_range=(115, 170),
    crack_intensity_range=(350, 550),
    noise_stddev=5,
    min_cracks=5,
    max_cracks=15,
    crack_zone_width=8,  # き裂描画可能区域の横幅
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)
    
    images_dir = Path(save_path) / "images"
    masks_dir = Path(save_path) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    images_generated = 0
    
    # 指定された数の画像が生成されるまで繰り返す
    while images_generated < num_images:
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
        
        # き裂描画可能区域を生成
        crack_zones = generate_crack_zones(material_mask, num_cracks, crack_zone_width)
        
        for zone in crack_zones:
            # ゾーン内の素材上端と下端を見つける
            x_center = zone['x_center']
            half_width = zone['half_width']
            
            # 素材マスクからこのゾーンに関連する列を抽出
            zone_x_min = max(0, x_center - half_width)
            zone_x_max = min(width - 1, x_center + half_width)
            
            # このゾーン内の素材の上端と下端を探す
            zone_mask = material_mask[:, zone_x_min:zone_x_max+1]
            zone_mask_any = np.any(zone_mask > 0.5, axis=1)
            
            if np.any(zone_mask_any):
                y_indices = np.where(zone_mask_any)[0]
                y_top = y_indices[0]
                y_bottom = y_indices[-1]
                
                if y_bottom - y_top < 20:  # 素材高さが小さすぎる場合はスキップ
                    continue
                
                # 3つの中間点のy座標を生成（条件: y_top < p1 < p2 < p3 < y_bottom）
                available_height = y_bottom - y_top - 4  # 各点に少なくとも1pxの間隔を確保
                
                if available_height < 6:  # 十分な高さがない場合はスキップ
                    continue
                
                # 高さを4つのセグメントに分割し、各セグメント内でランダムに点を配置
                segment_height = available_height / 4
                
                # 各セグメントの境界
                boundaries = [
                    y_top + 1,  # セグメント1の開始
                    y_top + 1 + int(segment_height),  # セグメント1の終了/セグメント2の開始
                    y_top + 1 + int(2 * segment_height),  # セグメント2の終了/セグメント3の開始
                    y_top + 1 + int(3 * segment_height),  # セグメント3の終了/セグメント4の開始
                    y_bottom - 1  # セグメント4の終了
                ]
                
                # 各セグメント内でランダムに点を生成
                p1_y = np.random.randint(boundaries[0], boundaries[1] + 1)
                p2_y = np.random.randint(boundaries[1] + 1, boundaries[2] + 1)
                p3_y = np.random.randint(boundaries[2] + 1, boundaries[3] + 1)
                
                # 各点のx座標をき裂描画可能区域内でランダムに生成
                x_top = np.random.randint(zone_x_min, zone_x_max + 1)
                x_p1 = np.random.randint(zone_x_min, zone_x_max + 1)
                x_p2 = np.random.randint(zone_x_min, zone_x_max + 1)
                x_p3 = np.random.randint(zone_x_min, zone_x_max + 1)
                x_bottom = np.random.randint(zone_x_min, zone_x_max + 1)
                
                # 5点のリストを作成（上端、3つの中間点、下端）
                points = [
                    (x_top, y_top),
                    (x_p1, p1_y),
                    (x_p2, p2_y),
                    (x_p3, p3_y),
                    (x_bottom, y_bottom)
                ]
                
                # 上端下端を含めて、隣接する点同士の太さの差が1以下になるようにする
                # 上端は1px固定
                thickness_top = 1
                
                # 上端から順に、隣接する点との太さの差が1px以下になるように設定
                thickness_p1 = max(crack_thickness_range[0], min(crack_thickness_range[1], 
                                 thickness_top + np.random.choice([-1, 0, 1])))
                thickness_p2 = max(crack_thickness_range[0], min(crack_thickness_range[1], 
                                 thickness_p1 + np.random.choice([-1, 0, 1])))
                thickness_p3 = max(crack_thickness_range[0], min(crack_thickness_range[1], 
                                 thickness_p2 + np.random.choice([-1, 0, 1])))
                
                # 下端は1px固定だが、p3との差が1px以内になるように調整する必要がある
                thickness_bottom = 1
                
                # p3との差が1px以上になる場合は、p3の太さを調整
                if abs(thickness_p3 - thickness_bottom) > 1:
                    # p3とbottomの差を1pxに調整
                    if thickness_p3 > thickness_bottom + 1:
                        thickness_p3 = thickness_bottom + 1  # 2
                    else:
                        # この場合はありえない（bottomが1px固定で、p3は常に1以上）
                        pass
                
                # 太さのリスト
                thicknesses = [thickness_top, thickness_p1, thickness_p2, thickness_p3, thickness_bottom]
                
                # ひび割れの明度をランダムに生成
                crack_intensity = sample_decreasing_probability(crack_intensity_range[0], crack_intensity_range[1])
                
                # ひび割れを描画（点を順番に結ぶ）
                for i in range(len(points) - 1):
                    pt1 = points[i]
                    pt2 = points[i + 1]
                    thickness = thicknesses[i]
                    
                    # 点間の距離に基づいて明度変化
                    dist = np.abs(pt2[1] - pt1[1])
                    intensity_factor = min(1.0, dist / 30.0)  # 距離が長いほど明るく
                    current_intensity = int(crack_intensity * (0.7 + 0.3 * intensity_factor))
                    
                    # ひび割れを描画
                    cv2.line(image, pt1, pt2, color=current_intensity, thickness=thickness)
                    cv2.line(crack_mask, pt1, pt2, color=255, thickness=thickness)
        
        # RAWファイルとして保存
        image_path = images_dir / f"{images_generated:04d}_SC.raw"
        image.astype(np.uint16).tofile(image_path)
        
        # マスクを保存
        mask_path = masks_dir / f"{images_generated:04d}_gt.png"
        cv2.imwrite(str(mask_path), crack_mask)
        
        # 生成した画像をカウント
        images_generated += 1
    
    return f"Saved {num_images} synthetic images with masks to {save_path}"

def generate_crack_zones(material_mask, num_zones, zone_width):
    """
    き裂描画可能区域をランダムに生成する関数
    """
    height, width = material_mask.shape
    zones = []
    
    for _ in range(num_zones):
        # ランダムなx座標を選択
        x_center = np.random.randint(0, width)
        
        half_width = zone_width // 2
        
        # x座標がゾーンの幅の半分より小さい場合は調整
        x_center = max(half_width, min(width - half_width - 1, x_center))
        
        zones.append({
            'x_center': x_center,
            'half_width': half_width
        })
    
    return zones

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
    
    # パラメータ設定 (修正された値)
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
    result = generate_synthetic_crack_image(
        num_images=100, 
        save_path="./synthetic_crack_dataset3",
        material_intensity_range=(250, 320),
        void_intensity_range=(115, 170),
        crack_intensity_range=(350, 600),
        crack_thickness_range=(1, 3),
        crack_zone_width=8
    )
    print(result)