import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib import font_manager, rc

# 日本語フォントの設定（フォントファイルがある場合）
def set_japanese_font():
    try:
        # macOSの場合
        if os.name == 'posix' and os.uname().sysname == 'Darwin':
            font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
            if os.path.exists(font_path):
                font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = 'Hiragino Sans'
                return True
        # Windowsの場合
        elif os.name == 'nt':
            font_path = 'C:/Windows/Fonts/meiryo.ttc'
            if os.path.exists(font_path):
                font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = 'Meiryo'
                return True
        # Linuxの場合
        elif os.name == 'posix':
            font_paths = [
                '/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf',
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
            ]
            for path in font_paths:
                if os.path.exists(path):
                    font_manager.fontManager.addfont(path)
                    plt.rcParams['font.family'] = path.split('/')[-1].split('.')[0]
                    return True
        return False
    except Exception as e:
        print(f"フォント設定エラー: {e}")
        return False

# 画像の正規化関数
def normalize(data, output_type=np.uint8):
    """
    画像データを0-255の範囲に正規化する関数
    
    引数:
        data: 正規化する画像データ
        output_type: 出力データ型（デフォルト: np.uint8）
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    # 最小値が最大値と同じ場合（一定値の画像）
    if data_min == data_max:
        return np.zeros(data.shape, dtype=output_type)
    
    # 0-255または0-65535に正規化
    max_value = 255 if output_type == np.uint8 else 65535
    normalized = ((data - data_min) / (data_max - data_min) * max_value).astype(output_type)
    
    return normalized

def load_raw_image(file_path, h, w):
    with open(file_path, "rb") as f:
        rawdata = f.read()
        data = np.frombuffer(rawdata, dtype=np.int16).reshape(h, w)
    return data

def create_histogram(image_data, bins=256, title="Brightness Histogram"):
    """
    画像データから輝度値のヒストグラムを作成する関数
    
    引数:
        image_data: 画像データ（numpy配列）
        bins: ヒストグラムのビン数（デフォルト: 256）
        title: ヒストグラムのタイトル
    """
    # データの最小値と最大値を取得
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    
    # ヒストグラムの生成
    plt.figure(figsize=(10, 6))
    hist, bins, _ = plt.hist(image_data.flatten(), bins=bins, color='blue', alpha=0.7)
    
    # グラフの設定
    plt.title(title)
    plt.xlabel('Brightness Value')
    plt.ylabel('Pixel Count')
    plt.grid(True, alpha=0.3)
    
    # 統計情報をテキストとして表示
    mean_val = np.mean(image_data)
    median_val = np.median(image_data)
    std_val = np.std(image_data)
    
    stats_text = f"Min: {min_val}\nMax: {max_val}\nMean: {mean_val:.2f}\nMedian: {median_val}\nStd Dev: {std_val:.2f}"
    plt.figtext(0.15, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.5))
    
    return hist, bins

def save_histogram_image(image_data, file_path, bins=256, title="Brightness Histogram"):
    """
    ヒストグラム画像をファイルに保存する関数
    """
    # プロット作成
    plt.figure(figsize=(10, 6))
    hist, bins, _ = plt.hist(image_data.flatten(), bins=bins, color='blue', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Brightness Value')
    plt.ylabel('Pixel Count')
    plt.grid(True, alpha=0.3)
    
    # 統計情報をテキストとして表示
    mean_val = np.mean(image_data)
    median_val = np.median(image_data)
    std_val = np.std(image_data)
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    
    stats_text = f"Min: {min_val}\nMax: {max_val}\nMean: {mean_val:.2f}\nMedian: {median_val}\nStd Dev: {std_val:.2f}"
    plt.figtext(0.15, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    print(f"ヒストグラム画像を保存しました: {file_path}")

def calculate_row_averages(image_data):
    """
    画像の各行（y座標）ごとの輝度値平均を計算する関数
    
    引数:
        image_data: 画像データ（numpy配列）
        
    戻り値:
        row_averages: 各行の平均輝度値を格納した配列
    """
    # 行ごとの平均値を計算（axis=1で行方向）
    row_averages = np.mean(image_data, axis=1)
    return row_averages

def plot_row_averages(row_averages, title="Row Average Brightness"):
    """
    行ごとの平均輝度値をプロットする関数
    
    引数:
        row_averages: 各行の平均輝度値を格納した配列
        title: グラフのタイトル
    """
    plt.figure(figsize=(10, 6))
    plt.plot(row_averages, 'b-')
    plt.title(title)
    plt.xlabel('Y-coordinate (Row)')
    plt.ylabel('Average Brightness')
    plt.grid(True, alpha=0.3)
    
    # 統計情報をテキストとして表示
    mean_val = np.mean(row_averages)
    median_val = np.median(row_averages)
    std_val = np.std(row_averages)
    min_val = np.min(row_averages)
    max_val = np.max(row_averages)
    
    stats_text = f"Min: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd Dev: {std_val:.2f}"
    plt.figtext(0.15, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    return plt.gcf()

def save_row_averages_plot(row_averages, file_path, title="Row Average Brightness"):
    """
    行ごとの平均輝度値プロットを保存する関数
    """
    fig = plot_row_averages(row_averages, title)
    plt.savefig(file_path)
    plt.close(fig)
    print(f"行平均プロットを保存しました: {file_path}")

def analyze_image_with_histogram(file_path, height, width, save_dir=None):
    """
    RAW画像を読み込み、y座標ごとの行平均輝度値に基づくヒストグラムを表示する関数
    
    引数:
        file_path: RAW画像ファイルのパス
        height: 画像の高さ（ピクセル）
        width: 画像の幅（ピクセル）
        save_dir: 結果を保存するディレクトリ（省略可）
    """
    # RAW画像の読み込み
    print(f"画像を読み込み中: {os.path.basename(file_path)}")
    image_data = load_raw_image(file_path, height, width)
    
    # 画像の基本情報を表示
    print(f"画像サイズ: {width}x{height}")
    print(f"データ型: {image_data.dtype}")
    print(f"輝度値範囲: {np.min(image_data)} 〜 {np.max(image_data)}")
    
    # 行ごとの平均輝度値を計算
    row_averages = calculate_row_averages(image_data)
    print(f"行平均輝度値範囲: {np.min(row_averages):.2f} 〜 {np.max(row_averages):.2f}")
    
    # 結果を保存する場合、ディレクトリを作成
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 画像名（拡張子なし）を取得
        base_name = os.path.basename(file_path).split(".")[0]
        
        # 正規化してPNG画像として保存
        norm_data = normalize(image_data)
        png_path = os.path.join(save_dir, f"{base_name}.png")
        cv2.imwrite(png_path, norm_data)
        print(f"正規化した画像を保存しました: {png_path}")
        
        # 行平均のヒストグラム画像を保存
        hist_path = os.path.join(save_dir, f"{base_name}_row_histogram.png")
        save_histogram_image(row_averages, hist_path, title=f"Row Average Histogram - {base_name}")
        
        # 行平均のプロット画像を保存
        plot_path = os.path.join(save_dir, f"{base_name}_row_plot.png")
        save_row_averages_plot(row_averages, plot_path, title=f"Row Average Plot - {base_name}")
    
    # 行平均輝度値のヒストグラムの作成と表示
    hist, bins = create_histogram(row_averages, title=f"Row Average Histogram - {os.path.basename(file_path)}")
    
    # 行平均輝度値のプロットを表示
    plot_row_averages(row_averages, title=f"Row Average Plot - {os.path.basename(file_path)}")
    
    # 元の画像も表示して比較できるようにする
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data, cmap='gray')
    plt.colorbar(label='Brightness')
    plt.title(f"Image - {os.path.basename(file_path)}")
    
    # 行平均の値を可視化した画像を作成・表示
    row_vis = np.zeros_like(image_data)
    for i, avg in enumerate(row_averages):
        row_vis[i, :] = int(avg)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(row_vis, cmap='viridis')
    plt.colorbar(label='Row Average Brightness')
    plt.title(f"Row Average Visualization - {os.path.basename(file_path)}")
    
    plt.tight_layout()
    plt.show()
    
    return image_data, row_averages, hist, bins

# 使用例
if __name__ == "__main__":
    # 日本語フォントの設定を試みる
    if not set_japanese_font():
        print("日本語フォントを設定できませんでした。英語でラベルを表示します。")
    
    # ファイルパスと画像サイズを指定
    file_path = "hyouka/2400_1_SC.raw"  # RAW画像ファイルのパスを指定
    height = 175  # 画像の高さを指定
    width = 638   # 画像の幅を指定
    
    # 結果を保存するディレクトリ（省略可）
    save_dir = "output"
    
    # 画像の分析を実行
    image_data, row_averages, hist, bins = analyze_image_with_histogram(file_path, height, width, save_dir)
    
    # 必要に応じて追加の分析もできます
    # 例：特定の輝度値範囲の行をカウント
    lower_bound = 1000
    upper_bound = 2000
    row_count = np.sum((row_averages >= lower_bound) & (row_averages <= upper_bound))
    total_rows = height
    percentage = (row_count / total_rows) * 100
    
    print(f"平均輝度値が {lower_bound}〜{upper_bound} の行数: {row_count}")
    print(f"全体に対する割合: {percentage:.2f}%")
    
    # 特定の輝度値以上または以下の行を検出
    threshold = np.mean(row_averages)  # 平均値をしきい値として使用
    bright_rows = np.where(row_averages > threshold)[0]
    dark_rows = np.where(row_averages <= threshold)[0]
    
    print(f"平均より明るい行: {len(bright_rows)} 行")
    if len(bright_rows) > 0:
        print(f"  先頭の5つ: {bright_rows[:5]}")
    
    print(f"平均より暗い行: {len(dark_rows)} 行")
    if len(dark_rows) > 0:
        print(f"  先頭の5つ: {dark_rows[:5]}")