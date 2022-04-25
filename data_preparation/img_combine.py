# *encoding=utf-8
import os
import re
import cv2

import utils


root_dir = "../Image-Downloader-master/download_images/gan/"  # emoji_all"
dir_list = ["emoji", "表情", "表情包", "斗图", "群表情"]
output_dir = os.path.join(root_dir, "emoji_combine")
utils.check_dir(output_dir)
size = (256, 256)

file_suffix = "jpeg|jpg|png"
index = 0
for dir_name in dir_list:
    index += 1
    dir_path = os.path.join(root_dir, dir_name)
    file_list = os.listdir(dir_path)
    for img_name in file_list:
        # 对处理文件的类型进行过滤
        if re.search(file_suffix, img_name) is None:
            continue
        img_path = dir_path + "/" + img_name
        print(img_path)
        try:
            img = utils.cv_imread(img_path)
        except:
            continue
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        if img_height >= size[1] and img_width >= size[0]:
            img = cv2.resize(img, size)

        output_path = os.path.join(output_dir, str(index) + "_" + img_name)
        cv2.imwrite(output_path, img)
