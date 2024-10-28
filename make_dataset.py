from PIL import Image as Image
import numpy as np
import glob
import re
import os
from tqdm import tqdm
import shutil
import cv2


# # move px every 1px
def cropped_center_pixel(input_dir, input_GT_dir, save_path, grid_size):
    grid_size = int(grid_size)
    GT_files = sorted(glob.glob(f"{input_GT_dir}/*"))
    png_files = sorted(glob.glob(f"{input_dir}/*"))
    try:
        shutil.rmtree(save_path)
    except:
        pass

    os.makedirs(f"{save_path}/train/crack")
    os.makedirs(f"{save_path}/validation/crack")
    os.mkdir(f"{save_path}/train/no_crack")
    os.makedirs(f"{save_path}/validation/no_crack")
    # for file in GT_files:
        #-------------------
        # rename = re.search("\w+/\w+", file).span()
        # power = file[rename[1]-7:rename[1]-3]
        
        # for png_file in png_files:
        #     if power in png_file:
        #         png_img = Image.open(png_file)
        #         pngimg_np = np.asarray(png_img)
        #         img = Image.open(file)
        #         img_np = np.asarray(img)
        #         y, x = img_np.shape
        #         # img_av = np.sum(img_np)/(y*x)
        #         pad = np.full((y+(grid_size//2*2), x+(grid_size//2*2)), 0)
        #         pad[grid_size//2:(grid_size//2)+y, grid_size//2:(grid_size//2)+x] = pngimg_np
        #         for i in range(grid_size//2, y+(grid_size//2)):
        #             for j in range(grid_size//2, x+(grid_size//2)):
        #                 save_data = pad[i-(grid_size//2):i+(grid_size//2)+1, j-(grid_size//2):j+(grid_size//2)+1]
        #                 if img_np[i-grid_size//2][j-grid_size//2] == 255:
        #                     Image.fromarray(np.uint8(save_data)).save(f"./{save_path}/train/crack/{power}__{j-grid_size//2}-{i-grid_size//2}.png")
        #                 else:
        #                     Image.fromarray(np.uint8(save_data)).save(f"./{save_path}/train/no_crack/{power}__{j-grid_size//2}-{i-grid_size//2}.png")
        #     else:
        #         continue
#----------------------
    for raw_file in tqdm(png_files):
            # "1800"または"1800_UL"のようなファイル名
            name = re.search("\w+/(.+)_SC_.+\.png", raw_file).group(1)
            try:
                raw = cv2.imread(raw_file, cv2.IMREAD_GRAYSCALE)
                gt = cv2.imread(f"{input_GT_dir}/{name}_re.png", cv2.IMREAD_GRAYSCALE)
                y, x = gt.shape
                pad = np.pad(raw, grid_size // 2, "constant")
                for i in range(grid_size // 2, y + (grid_size // 2)):
                    for j in range(grid_size // 2, x + (grid_size // 2)):
                        save_data = pad[
                            i - (grid_size // 2) : i + (grid_size // 2) + 1,
                            j - (grid_size // 2) : j + (grid_size // 2) + 1,
                        ]
                        if gt[i - grid_size // 2][j - grid_size // 2] == 255:
                            cv2.imwrite(
                                f"./{save_path}/train/crack/{name}__{j-grid_size//2}-{i-grid_size//2}.png",
                                save_data,
                            )

                        else:
                            cv2.imwrite(
                                f"./{save_path}/train/no_crack/{name}__{j-grid_size//2}-{i-grid_size//2}.png",
                                save_data,
                            )

            except Exception as e:
                print(f"cropped_center_pixelで{e}発生")

#----------------------------------------------------------------------------------------------------
# # move px every grid size
def cropped_area_size(input_dir, input_GT_dir, save_path, grid_size):
    grid_size = int(grid_size)
    files = sorted(glob.glob(f"{input_GT_dir}/*"))
    png_files = sorted(glob.glob(f"{input_dir}/*"))
    try:
        shutil.rmtree(save_path)
    except:
        pass

    os.makedirs(f"{save_path}/train/crack")
    os.makedirs(f"{save_path}/validation/crack")
    os.mkdir(f"{save_path}/train/no_crack")
    os.makedirs(f"{save_path}/validation/no_crack")
    #---------------
    # for file in files:
    #     rename = re.search("\w+/\w+", file).span()
    #     # print(file[rename[1]-4:rename[1]])
    #     power = file[rename[1]-7:rename[1]-3]
    #     for png_file in png_files:
    #         if power in png_file:
    #             png_img = Image.open(png_file)
    #             pngimg_np = np.asarray(png_img)
    #             img = Image.open(file)
    #             img_np = np.asarray(img)
    #             y, x = img_np.shape
    #             for i in range(y//grid_size):
    #                 for j in range(x//grid_size):
    #                     position_x = j*grid_size
    #                     position_y = i*grid_size
    #                     crop_data = img_np[position_y:position_y+grid_size, position_x:position_x+grid_size]
    #                     # save_data = pngimg_np[position_y:position_y+grid_size, position_x:position_x+grid_size]
    #                     save_data = pngimg_np[position_y:position_y+grid_size, position_x:position_x+grid_size]
    #                     # Image.fromarray(np.uint8(save_data)).save(f"./2class_Original_data/test/fracture/{power}__{position_x}-{position_y}.png")
    #                     if 255 in crop_data:

    #                         Image.fromarray(np.uint8(save_data)).save(f"./{save_path}/train/crack/{power}__{position_x}-{position_y}.png")
    #                     else:
    #                         Image.fromarray(np.uint8(save_data)).save(f"./{save_path}/train/no_crack/{power}__{position_x}-{position_y}.png")
    #         else:
    #             continue
            #-------------------------
    for raw_file in tqdm(png_files):
        # "1800"または"1800_UL"のようなファイル名
        name = re.search("\w+/(.+)_SC_.+\.png", raw_file).group(1)
        try:
            raw = cv2.imread(raw_file, cv2.IMREAD_GRAYSCALE)
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