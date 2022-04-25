import os
import pandas as pd
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

import utils


root_dir = "../Image-Downloader-master/download_images/gan/emoji_combine"
excel_path = "img_combine_info_type.xls"
data = pd.read_excel(excel_path, index_col=0)

columns = ["size", "area", "gradient", "si", "niqe", "colorful"]
color = {"lr": "red", "hr": "blue", "other": "yellow"}
for column in columns:
    for type_now in color.keys():
        data_type = data[data["type"] == type_now].drop("type", axis=1)
        print(type_now, data_type)
        if len(data_type) == 0:
            continue
        utils.draw_hist(data_type[column], type_now, column, color[type_now])
    plt.show()

# sample_weight = {"hr": 4, "lr": 2, "other": 1}
clf = DecisionTreeClassifier(max_depth=3)  # 初始化
clf = clf.fit(data[columns], data["type"])  # 拟合
joblib.dump(clf, 'clf.model')

os.environ["PATH"] += os.pathsep + 'd:/Program Files/Graphviz/bin/'
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=columns)
graph = graphviz.Source(dot_data)
graph.render("DecisionTree")

