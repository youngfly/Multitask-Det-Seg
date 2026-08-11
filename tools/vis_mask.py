# coding: utf-8
import os
import os.path
import xml.etree.cElementTree as ET
import cv2
import numpy as np
import json

images = '../dataset/dataset_extra/png'
anno = '../dataset/dataset_extra/png_seg/'
vis_path = '../dataset/dataset_extra/vis_mask/'

for i, img in enumerate(os.listdir(images)):
    im = cv2.imread(images+'/'+img)
    mask_path = anno + img.split('.')[0] + '_seg.png'
    seg_label = cv2.imread(mask_path, 0)

    color_area = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
    color_area[seg_label > 1] = [255, 255, 0]
    color_seg = color_area
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    im[color_mask != 0] = im[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    im = im.astype(np.uint8)
    cv2.imwrite(vis_path + img + '.jpg', im)

