import os
import cv2
import pygame
import pandas as pd
import matplotlib.pyplot as plt

import utils


def show_next_img(i):
    name = data.index[i]
    file_path = os.path.join(root_dir, name)
    print(file_path)
    img = pygame.image.load(file_path)
    img = pygame.transform.scale(img, show_size)
    screen.blit(img, (0, 0))
    pygame.display.flip()


root_dir = "../Image-Downloader-master/download_images/gan/emoji_combine"
excel_path = "img_combine_info.xls"
data = pd.read_excel(excel_path, index_col=0)
data = data.sample(50)
data["type"] = ""
show_size = (256, 256)

pygame.init()
screen = pygame.display.set_mode(show_size)

i = 0
show_next_img(i)
end_tag = False
type_dict = {pygame.K_LEFT: "lr", pygame.K_RIGHT: "hr", pygame.K_UP: "other"}
while not end_tag:
    for event in pygame.event.get():
        print(event.type)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pygame.quit()
                end_tag = True
                break
            if event.key in type_dict.keys():
                data.loc[data.index[i], "type"] = type_dict[event.key]
                i += 1
                if i >= len(data):
                    pygame.quit()
                    end_tag = True
                    break
                show_next_img(i)

writer = pd.ExcelWriter("img_combine_info_type.xls")
data.to_excel(writer, index=True, float_format="%.f")
writer.save()

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
