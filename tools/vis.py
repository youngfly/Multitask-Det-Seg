# coding: utf-8
import os
import os.path
import xml.etree.cElementTree as ET
import cv2
import numpy as np
import json

images = '../dataset/dataset_extra/png'
anno = '../dataset/dataset_extra/json_det/'
vis_path = '../dataset/dataset_extra/vis/'

for i, img in enumerate(os.listdir(images)):
    im = cv2.imread(images+'/'+img)
    label_path = anno + img.split('.')[0] + '_det.json'
    with open(label_path, 'r') as f:
        label = json.load(f)
    data = label['objects']
    for idx, obj in enumerate(data):
        category = obj['category']
        x1 = float(obj['box2d']['x1'])
        y1 = float(obj['box2d']['y1'])
        x2 = float(obj['box2d']['x2'])
        y2 = float(obj['box2d']['y2'])
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
        color = (0, 255, 0)
        cv2.rectangle(im, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
        # cv2.rectangle(im, c1, c2, color, thickness=1)
    cv2.imwrite(vis_path + img+'.jpg', im)


