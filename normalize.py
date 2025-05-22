# import numpy as np

# def load_raw_image(file_path, h, w):
#     """
#     RAW画像をnumpy配列として読み込む関数。
#     """
#     with open(file_path, "rb") as f:
#         raw_data = f.read()
#         image = np.frombuffer(raw_data, dtype=np.int16).reshape(h, w)
#     return image

# def normalize_raw_background(image):
#     """
#     RAW画像の背景輝度を正規化する関数。
#     上部と下部の5ピクセル行の平均明度を計算し、その平均値を画像全体から引く。
#     結果が0未満になる場合は0に設定する。
    
#     Args:
#         image (numpy.ndarray): 入力RAW画像のnumpy配列
        
#     Returns:
#         numpy.ndarray: 背景輝度正規化後の画像
#     """
#     # 画像の上部5行と下部5行を取得
#     top_rows = image[:5, :]
#     bottom_rows = image[-5:, :]
    
#     # 上部と下部の平均明度を計算
#     top_mean = np.mean(top_rows)
#     bottom_mean = np.mean(bottom_rows)
    
#     # 全体の平均明度を計算
#     background_mean = (top_mean + bottom_mean) / 2
    
#     # 正規化した画像を作成（バックグラウンド輝度を引く）
#     normalized_image = image - background_mean
    
#     # 負の値を0に置換
#     normalized_image = np.maximum(normalized_image, 0)
    
#     return normalized_image

# def process_raw_image(file_path, h, w, output_path=None):
#     """
#     RAW画像を読み込み、背景輝度を正規化して保存する。
    
#     Args:
#         file_path (str): 入力RAW画像のパス
#         h (int): 画像の高さ
#         w (int): 画像の幅
#         output_path (str, optional): 出力ファイルパス。Noneの場合は保存しない。
        
#     Returns:
#         numpy.ndarray: 正規化された画像
#     """
#     # RAW画像の読み込み
#     raw_image = load_raw_image(file_path, h, w)
    
#     # 背景輝度の正規化
#     normalized_image = normalize_raw_background(raw_image)
    
#     # 結果を保存する場合
#     if output_path:
#         with open(output_path, "wb") as f:
#             f.write(normalized_image.astype(np.int16).tobytes())
    
#     return normalized_image

# # 処理して結果を保存する
# process_raw_image("hyouka/2400_1_SC.raw", 175, 638, output_path="normalized.raw")

import numpy as np

def load_raw_image(file_path, h, w):
    """
    RAW画像をnumpy配列として読み込む関数。
    
    Args:
        file_path (str): 入力RAW画像のパス
        h (int): 画像の高さ
        w (int): 画像の幅
        
    Returns:
        numpy.ndarray: 読み込まれたRAW画像のnumpy配列
    """
    with open(file_path, "rb") as f:
        raw_data = f.read()
        image = np.frombuffer(raw_data, dtype=np.int16).reshape(h, w)
    return image

def normalize_background(input_path, h, w, output_path=None):
    """
    RAW画像の背景輝度を正規化する関数。
    上部と下部の5ピクセル行の平均明度を計算し、その平均値を画像全体から引く。
    結果が0未満になる場合は0に設定する。
    
    Args:
        image (numpy.ndarray): 入力RAW画像のnumpy配列
        output_path (str, optional): 出力ファイルパス。Noneの場合は保存しない。
        
    Returns:
        numpy.ndarray: 背景輝度正規化後の画像
    """
    
    image = load_raw_image(input_path, h, w)
    # 画像の上部5行と下部5行を取得
    top_rows = image[:5, :]
    bottom_rows = image[-5:, :]
    
    # 上部と下部の平均明度を計算
    top_mean = np.mean(top_rows)
    bottom_mean = np.mean(bottom_rows)
    
    # 全体の平均明度を計算
    background_mean = (top_mean + bottom_mean) / 2
    
    # 正規化した画像を作成（バックグラウンド輝度を引く）
    normalized_image = image - background_mean
    
    # 負の値を0に置換
    normalized_image = np.maximum(normalized_image, 0)
    
    with open(output_path, "wb") as f:
        f.write(normalized_image.astype(np.int16).tobytes())
    

normalize_background("hyouka/2400_1_SC.raw", 175, 638, "normalized.raw")