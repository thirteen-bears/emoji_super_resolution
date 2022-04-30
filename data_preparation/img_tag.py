import os
import cv2
import pygame
import pandas as pd
import matplotlib.pyplot as plt

import utils

os.environ['QT_MAC_WANTS_LAYER'] = '1'


def show_next_img(i):
    name = data.index[i]
    file_path = os.path.join(root_dir, name)
    print(file_path)
    img = pygame.image.load(file_path)
    img = pygame.transform.scale(img, show_size)
    screen.blit(img, (0, 0))
    pygame.display.flip()


root_dir = "../../emoji_combine"
excel_path = "img_combine_info.xls"
data = pd.read_excel(excel_path, index_col=0)
data = data.sample(50)
data["type"] = "" # add a new column "tag""
show_size = (256, 256) # pygame screen resolution

pygame.init()
screen = pygame.display.set_mode(show_size)

i = 0
show_next_img(i)
end_tag = False # whether to end the project
type_dict = {pygame.K_LEFT: "lr", pygame.K_RIGHT: "hr", pygame.K_UP: "other"} # key and corresponding tag
while not end_tag:
    for event in pygame.event.get(): # get event from q 监听键盘的事件
        #print(event.type)
        if event.type == pygame.KEYDOWN: # keydown 按下键盘某个按键
            if event.key == pygame.K_SPACE:  # key space means quit
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
data.to_excel(writer, index=True, float_format="%.f") # write "data" into excel
writer.save()

# visualize the features
columns = ["size", "area", "gradient", "si", "niqe", "colorful"]
color = {"lr": "red", "hr":  "blue", "other": "yellow"}
for column in columns:
    for type_now in color.keys(): 
        data_type = data[data["type"] == type_now].drop("type", axis=1) #get one and drop one
        print(type_now, data_type)
        if len(data_type) == 0:
            continue
        utils.draw_hist(data_type[column], type_now, column, color[type_now])
    plt.show()
