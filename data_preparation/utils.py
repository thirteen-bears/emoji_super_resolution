import os
import re
import cv2
import seaborn
import numpy as np
import matplotlib.pyplot as plt


def get_file_list(input_path, extension="jpeg|jpg|png"):
    if os.path.isdir(input_path):
        file_list = []
        for root, dirs, files in os.walk(input_path):
            print("root = ", root)
            for file in files:
                file_list.append(os.path.join(root, file))
    else:
        file_list = [input_path]
    if extension:
        file_list = [f for f in file_list if re.search(extension, f) is not None]
    return file_list


def get_file_name(file_path, extension=True):
    file_name = os.path.split(str(file_path))[-1]
    if not extension:
        file_name = os.path.splitext(file_name)[0]
    return file_name


def check_dir(dir_path):
    path_list = dir_path.split("/")
    now_path = ""
    for path in path_list:
        now_path += path + "/"
        if not os.path.exists(now_path):
            os.makedirs(now_path)


def draw_hist(score, title="", x_label="", facecolor='lightskyblue'):
    plt.hist(score, 30, histtype='bar', facecolor=facecolor, edgecolor="black", alpha=0.8, rwidth=1, density=True)
    plt.title(title)
    plt.xlabel(x_label)


def draw_KDE(new_score, title="", x_label=""):
    p1 = seaborn.kdeplot(new_score, color="red")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Probability density")


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img
