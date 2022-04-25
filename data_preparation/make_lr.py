import os
import cv2

import utils

root_dir = "../Image-Downloader-master/download_images/gan/validation"
input_dir = os.path.join(root_dir, "hr")
output_dir = os.path.join(root_dir, "lr")
new_size = (64, 64)
utils.check_dir(output_dir)

file_list = utils.get_file_list(root_dir)

for file_path in file_list:
    file_name = utils.get_file_name(file_path)
    img = cv2.imread(file_path)
    img = cv2.resize(img, new_size)

    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, img)
    print(output_path)
