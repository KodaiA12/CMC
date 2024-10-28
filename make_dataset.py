from PIL import Image as Image
import numpy as np
import glob
import re
import os
from tqdm import tqdm
import shutil
import cv2

# RAW画像を読み込む関数
def load_raw_image(file_path, h=1344, w=1344):
    """
    .rawファイルを読み込み、指定されたサイズにリシェイプする関数。
    
    Parameters:
    - file_path (str): rawファイルのパス
    - h (int): 画像の高さ
    - w (int): 画像の幅
    
    Returns:
    - data (numpy array): 画像データ
    """
    with open(file_path, "rb") as f:
        rawdata = f.read()
        data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
    return data

# 指定サイズごとにクロップする関数
def cropped_area_size(input_dir, input_GT_dir, save_path, grid_size):
    grid_size = int(grid_size)
    files = sorted(glob.glob(f"{input_GT_dir}/*"))
    raw_files = sorted(glob.glob(f"{input_dir}/*.raw"))
    
    try:
        shutil.rmtree(save_path)
    except:
        pass

    os.makedirs(f"{save_path}/train/crack")
    os.makedirs(f"{save_path}/validation/crack")
    os.mkdir(f"{save_path}/train/no_crack")
    os.makedirs(f"{save_path}/validation/no_crack")

    for raw_file in tqdm(raw_files):
        # "1800"または"1800_UL"のようなファイル名
        name = re.search("\w+/(.+)_SC_.+\.raw", raw_file).group(1)
        
        try:
            # RAW画像を読み込む
            raw = load_raw_image(raw_file)
            
            # 対応するPNGラベル画像を読み込む
            gt = cv2.imread(f"{input_GT_dir}/{name}_re.png", cv2.IMREAD_GRAYSCALE)
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
                    if 255 in c_gt:  # き裂を含む領域だった場合
                        cv2.imwrite(
                            f"./{save_path}/train/crack/{name}__{position_x}-{position_y}.png",
                            c_raw,
                        )
                    else:
                        cv2.imwrite(
                            f"./{save_path}/train/no_crack/{name}__{position_x}-{position_y}.png",
                            c_raw,
                        )
        except Exception as e:
            print(f"cropped_area_sizeで{e}発生")

# 使用例
cropped_area_size("test/data/first/1ex_cropped", "test/data/first/first_label", "test/data/first/dataset", 50)
