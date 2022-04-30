import os
import cv2
import shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

import utils


def move_data(data, root_dir, output_dir):
    for i in range(len(data)):
        file_name = data.index[i]
        print(file_name)
        file_path = os.path.join(root_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        shutil.copy(file_path, output_path)


root_dir = "../../data/emoji_combine"
hr_dir = "../../data/hr"
lr_dir = "../../data/lr"
other_dir = "../../data/other"

utils.check_dir(hr_dir)
utils.check_dir(lr_dir)
utils.check_dir(other_dir)

model = joblib.load('clf.model')

excel_path = "img_combine_info.xls"
data = pd.read_excel(excel_path, index_col=0)
data.dropna(inplace=True)
columns = ["size", "area", "gradient", "si", "niqe", "colorful"]
data["type"] = model.predict(data[columns])

temp = data["type"] == "hr"
hr = data[data["type"] == "hr"]
lr = data[data["type"] == "lr"]
other = data[data["type"] == "other"]
# hr = data[(data["gradient"] <= 91.5) | ((163 < data["gradient"]) & (data["gradient"] <= 183))]
hr = hr[(hr["width"] >= 256) & (hr["height"] >= 256)]
# other = data[(91.5 < data["gradient"]) & (data["gradient"] <= 163) & ((72.5 < data["si"]) & (data["si"] <= 82.5))]
# lr = data.drop(hr.index).drop(other.index)

move_data(hr, root_dir, hr_dir)
move_data(lr, root_dir, lr_dir)
move_data(other, root_dir, other_dir)
