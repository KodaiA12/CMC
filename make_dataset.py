# from PIL import Image as Image
# import numpy as np
# import glob
# import re
# import os
# from tqdm import tqdm
# import shutil
# import cv2

# # RAW画像を読み込む関数
# def load_raw_image(file_path, h=100, w=460):
#     """
#     .rawファイルを読み込み、指定されたサイズにリシェイプする関数。
#     """
#     try:
#         with open(file_path, "rb") as f:
#             rawdata = f.read()
#             data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
#         return data
#     except Exception as e:
#         print(f"Failed to load raw image {file_path}: {e}")
#         return None

# # 指定サイズごとにクロップする関数
# def cropped_area_size(input_dir, input_GT_dir, save_path, grid_size):
#     grid_size = int(grid_size)
#     raw_files = sorted(glob.glob(f"{input_dir}/*.raw"))
    
#     try:
#         shutil.rmtree(save_path)
#     except Exception as e:
#         print(f"Failed to delete save path {save_path}: {e}")

#     os.makedirs(f"{save_path}/train/crack", exist_ok=True)
#     os.makedirs(f"{save_path}/validation/crack", exist_ok=True)
#     os.makedirs(f"{save_path}/train/no_crack", exist_ok=True)
#     os.makedirs(f"{save_path}/validation/no_crack", exist_ok=True)

#     for raw_file in tqdm(raw_files):
#         # "1800"または"1800_UL"のようなファイル名を抽出
#         name_match = re.search(r"\w+/(.+)_SC\.raw", raw_file)
        
#         if name_match:
#             name = name_match.group(1)
#         else:
#             print(f"File name pattern not matched for {raw_file}")
#             continue
        
#         # RAW画像を読み込む
#         raw = load_raw_image(raw_file)
        
#         # 読み込みエラーが発生した場合はスキップ
#         if raw is None:
#             continue
        
#         # 対応するPNGラベル画像を読み込む
#         gt_path = os.path.join(input_GT_dir, f"{os.path.basename(name)}.png")
#         gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
#         img_name = os.path.basename(name)

#         # ラベル画像が読み込めなかった場合はスキップ
#         if gt is None:
#             print(f"Label image not found or failed to load: {gt_path}")
#             continue

#         y, x = gt.shape

#         # 余りを計算し、その分を画像右下方向にパッド
#         y_surp, x_surp = y % grid_size, x % grid_size
#         raw_pad = np.pad(
#             raw, ((0, grid_size - y_surp), (0, grid_size - x_surp)), "constant"
#         )
#         gt_pad = np.pad(
#             gt, ((0, grid_size - y_surp), (0, grid_size - x_surp)), "constant"
#         )
#         y, x = raw_pad.shape

#         for i in range(y // grid_size):
#             for j in range(x // grid_size):
#                 position_x = j * grid_size
#                 position_y = i * grid_size
#                 c_raw = raw_pad[
#                     position_y : position_y + grid_size,
#                     position_x : position_x + grid_size,
#                 ]
#                 c_gt = gt_pad[
#                     position_y : position_y + grid_size,
#                     position_x : position_x + grid_size,
#                 ]
#                 if 255 in c_gt:  # き裂を含む領域だった場合
#                     with open(os.path.join(f"{save_path}/train/crack", f"{img_name}__{position_x}-{position_y}.raw"), "wb") as f:
#                         f.write(c_raw.tobytes())
#                 else:
#                     with open(os.path.join(f"{save_path}/train/no_crack", f"{img_name}__{position_x}-{position_y}.raw"), "wb") as f:
#                         f.write(c_raw.tobytes())

# # 使用例
# cropped_area_size("data/first/1ex_cropped", "data/first/first_label", "data/first/dataset", 50)
#-------------------------------------------------------------------------------------------------------------------
from PIL import Image as Image
import numpy as np
import glob
import re
import os
from tqdm import tqdm
import shutil
import cv2

# RAW画像を読み込む関数
def load_raw_image(file_path, h=100, w=460):
    try:
        with open(file_path, "rb") as f:
            rawdata = f.read()
            data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
        return data
    except Exception as e:
        print(f"Failed to load raw image {file_path}: {e}")
        return None

# 指定サイズごとにクロップし、7:3でtrainとvalidationに分割する関数
def cropped_area_size(input_dir, input_GT_dir, save_path, grid_size):
    grid_size = int(grid_size)
    raw_files = sorted(glob.glob(f"{input_dir}/*.raw"))

    # 対象フォルダのみ削除して再作成
    for sub_dir in ["image/train", "image/validation", "label/train", "label/validation"]:
        target_dir = os.path.join(save_path, sub_dir)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

    cropped_images = []  # クロップされた画像とラベルを一時的に保存するリスト

    for raw_file in tqdm(raw_files):
        # ファイル名を抽出（ディレクトリ情報を除去）
        name_match = re.search(r"(.+)_SC\.raw$", os.path.basename(raw_file))
        if name_match:
            name = name_match.group(1)
        else:
            print(f"File name pattern not matched for {raw_file}")
            continue

        # RAW画像を読み込む
        raw = load_raw_image(raw_file)

        if raw is None:
            continue

        # ラベル画像を読み込む
        gt_path = os.path.join(input_GT_dir, f"{name}.png")  # ファイル名から構築
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        img_name = os.path.basename(name)

        if gt is None:
            print(f"Label image not found or failed to load: {gt_path}")
            continue

        y, x = gt.shape

        # 余りを計算し、その分を画像右下方向にパッド
        y_surp, x_surp = y % grid_size, x % grid_size
        raw_pad = np.pad(
            raw, ((0, grid_size - y_surp), (0, grid_size - x_surp)), "constant"
        )
        gt_pad = np.pad(
            gt, ((0, grid_size - y_surp), (0, grid_size - x_surp)), "constant"
        )
        y, x = raw_pad.shape

        for i in range(y // grid_size):
            for j in range(x // grid_size):
                position_x = j * grid_size
                position_y = i * grid_size
                c_raw = raw_pad[
                    position_y : position_y + grid_size,
                    position_x : position_x + grid_size,
                ]
                c_gt = gt_pad[
                    position_y : position_y + grid_size,
                    position_x : position_x + grid_size,
                ]
                # ファイル名とデータをリストに追加
                cropped_images.append((img_name, position_x, position_y, c_raw, c_gt))

    # クロップされた画像を7:3に分割して保存
    total = len(cropped_images)
    train_count = int(total * 0.7)

    for idx, (img_name, position_x, position_y, c_raw, c_gt) in enumerate(cropped_images):
        # 保存先ディレクトリの決定
        if idx < train_count:
            img_save_dir = os.path.join(save_path, "image", "train")
            label_save_dir = os.path.join(save_path, "label", "train")
        else:
            img_save_dir = os.path.join(save_path, "image", "validation")
            label_save_dir = os.path.join(save_path, "label", "validation")

        # 元画像を保存
        img_file_name = f"{img_name}__{position_x}-{position_y}.raw"
        with open(os.path.join(img_save_dir, img_file_name), "wb") as f:
            f.write(c_raw.tobytes())

        # ラベル画像を保存
        label_file_name = f"{img_name}__{position_x}-{position_y}.png"
        cv2.imwrite(os.path.join(label_save_dir, label_file_name), c_gt)

# 使用例
cropped_area_size(
    input_dir="data/first/1ex_cropped",       # 元画像のディレクトリ
    input_GT_dir="data/first/first_label",   # ラベル画像のディレクトリ
    save_path="data/first",                  # 保存先のベースディレクトリ
    grid_size=32                             # クロップサイズ
)
