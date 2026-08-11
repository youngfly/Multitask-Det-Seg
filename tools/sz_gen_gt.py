# this file convert the color map to ground truth

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm

path = '../dataset/seglabels/train'
save_path = '../dataset/seglabels2/train'
if not os.path.exists(save_path):
    os.mkdir(save_path)
colormap_list = os.listdir(path)
print(colormap_list)

# suzhou-data
background = ([0, 0, 0], 0)
table = ([0,85,255], 1)  # （像素值， 标签）
ground = ([127, 127, 127], 2)
peomask = ([0,170,0], 3)
peonomask = ([255,85,0], 4)


class_lst = [ background, table, ground, peomask, peonomask]


table = {}
for key, idx in class_lst:
    table[tuple(key)] = idx
    table[tuple(key[:3])] = idx


print(table)

for colormap in tqdm.tqdm(colormap_list):

    print(colormap)
    img_colormap = np.array(Image.open(os.path.join(path, colormap)))
    print(img_colormap.shape)
    print(img_colormap[0, 0])
    shape = img_colormap.shape
    gt = np.ones((shape[0], shape[1]))*255
    for i in range(shape[0]):
        for j in range(shape[1]):
            if len(img_colormap[i, j]) == 3:
                key = tuple(img_colormap[i, j])
            else:
                key = tuple(img_colormap[i, j][:-1])
            # assert key in table, "not in table"
            try:
                gt[i, j] = table[key]
            except:
                gt[i, j] = 255
                print("{} is not in table at {},{}".format(key,i,j))

    gt_img = Image.fromarray(np.uint8(gt))
    gt_img.save(os.path.join(save_path, colormap.split('.')[0] + '.png'))
    print('finish {}'.format(colormap))
