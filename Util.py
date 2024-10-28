import numpy as np
import os
import cv2
import glob
import re


def change_fname(file_dir, pattern):
    path_list = glob.glob(f"{file_dir}/*")

    for file_path in path_list:
        try:
            dir, name = os.path.split(file_path)
            new_name = re.match(pattern, name).group(1)
            ext = os.path.splitext(file_path)[-1]  # 拡張子
            os.rename(file_path, dir + "/" + new_name + ext)
        except AttributeError:
            print(f"Error : {file_path}")


def crop_gt(file_path, save_dir, t_left=(400, 535), b_right=(815, 660)):
    """pngなどの，cv2.imread()が読み込める形式のGT画像を指定した位置で切り抜き，保存する

    Args:
        file_path (str): 読み込む画像ファイルのパス
        save_dir (str): 保存先のディレクトリ
        t_left (tuple, optional): 切り取り位置の左上座標. Defaults to (400, 535).
        b_right (tuple, optional): 切り取り位置の右下座標. Defaults to (815, 660).
    """
    x1, y1 = t_left
    x2, y2 = b_right

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    cropped_img = img[y1:y2, x1:x2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    name = os.path.basename(file_path)
    cv2.imwrite(os.path.join(save_dir, name), cropped_img)


def multi_crop_gt(file_dir, save_dir, t_left=(400, 535), b_right=(815, 660)):
    x1, y1 = t_left
    x2, y2 = b_right

    path_list = glob.glob(f"{file_dir}/*")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_path in path_list:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        cropped_img = img[y1:y2, x1:x2]

        name = os.path.basename(file_path)
        cv2.imwrite(os.path.join(save_dir, name), cropped_img)


def crop_raw(
    file_path, save_dir, w=1344, h=1344, t_left=(400, 535), b_right=(815, 660)
):
    x1, y1 = t_left
    x2, y2 = b_right

    # データの読み込み
    with open(file_path, "rb") as f:
        name = os.path.basename(file_path)
        rawdata = f.read()
        data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
        cropped_data = data[y1:y2, x1:x2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, name), "wb") as f:
        f.write(cropped_data.tobytes())


def multi_crop_raw(
    file_dir, save_dir, w=1344, h=1344, t_left=(400, 535), b_right=(815, 660)
):
    x1, y1 = t_left
    x2, y2 = b_right

    path_list = glob.glob(f"{file_dir}/*")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_path in path_list:
        # データの読み込み
        with open(file_path, "rb") as f:
            name = os.path.basename(file_path)
            rawdata = f.read()
            data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
            cropped_data = data[y1:y2, x1:x2]

        with open(os.path.join(save_dir, name), "wb") as f:
            f.write(cropped_data.tobytes())


def rotate_raw(file_path, save_dir, w=1344, h=1344, deg=15.5):
    # データの読み込み
    with open(file_path, "rb") as f:
        name = os.path.basename(file_path)
        rawdata = f.read()
        data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, name), "wb") as f:
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), int(deg), 1.0)
        rotated_data = cv2.warpAffine(data, mat, (w, h), flags=cv2.INTER_NEAREST)
        f.write(rotated_data.tobytes())


def multi_rotate_raw(file_dir, save_dir, w=1344, h=1344, deg=16):
    path_list = glob.glob(f"{file_dir}/*")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # データの読み込み
    for file_path in path_list:
        with open(file_path, "rb") as f:
            name = os.path.basename(file_path)
            rawdata = f.read()
            data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)

        with open(os.path.join(save_dir, name), "wb") as f:
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), int(deg), 1.0)
            rotated_data = cv2.warpAffine(data, mat, (w, h), flags=cv2.INTER_NEAREST)
            f.write(rotated_data.tobytes())


# 配列の正規化(画像として表示することが可能)
def normalize(array, box=None, value_range=None):
    """配列を正規化し，画像として表示することを可能にする

    Args:
        array (ndarray): 正規化したい配列
        box (tuple, optional): 注目領域の指定．この領域の最小最大値によって正規化される．順番は(左上, 右下)，形式は((y,x),(y,x)). Defaults to None.
        range (tuple, optional): 最小最大値の指定．この範囲で正規化される．boxも入力されている場合，そちらが優先される. Defaults to None.

    Returns:
        ndarray: 正規化された画像.
    """
    if box:
        tl, br = box
        _min = array[tl[0] : br[0], tl[1] : br[1]].min()
        _max = array[tl[0] : br[0], tl[1] : br[1]].max()
    elif value_range:
        _min, _max = value_range
    else:
        _min, _max = array.min(), array.max()

    normalized_data = ((array - _min) / (_max - _min)) * 255
    normalized_data = np.clip(normalized_data, 0, 255)

    return normalized_data.astype(np.uint8)


def raw2png(file_path, save_dir, w, h):
    # データの読み込み
    with open(file_path, "rb") as f:
        name = os.path.basename(file_path).split(".")[0]
        rawdata = f.read()
        data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    norm_data = normalize(data)
    cv2.imwrite(os.path.join(save_dir, name + ".png"), norm_data)


def multi_raw2png(file_dir, save_dir, w, h):
    path_list = glob.glob(f"{file_dir}/*")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file_path in path_list:
        # データの読み込み
        with open(file_path, "rb") as f:
            name = os.path.basename(file_path).split(".")[0]
            rawdata = f.read()
            data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)

        # norm_data = normalize(data, value_range=(1891, 2217))  # DPH
        # norm_data = normalize(data, value_range=(200, 455))  # AT
        # SC
        # norm_data = normalize(data, value_range=(254, 612))  # 1ex
        # norm_data = normalize(data, value_range=(120, 375))  # 3ex
        # norm_data = normalize(data, value_range=(120, 612))  # all

        # 通常
        norm_data = normalize(data)
        cv2.imwrite(os.path.join(save_dir, name + ".png"), norm_data)


if __name__ == "__main__":
    # 1ex
    multi_crop_raw("test/data/first/first_data", "test/data/first/1ex_cropped", t_left=(400, 550), b_right=(860, 650))
    # multi_raw2png("1ex_cropped", "1ex_png", 574, 130)

    # 3ex
    # multi_rotate_raw("data/3_ex", "3ex")
    # multi_crop_raw("3ex", "3ex", t_left=(341, 518), b_right=(973, 704))
    # multi_raw2png("3ex", "3ex", 632, 186)

    # change_fname("data_origin/1ex", "(\d+)")
