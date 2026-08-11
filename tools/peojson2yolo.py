
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from tqdm import tqdm
from os import getcwd
import json
import os

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    # try:
        with open(image_id, 'r') as f:
            label = json.load(f)
        # print(image_id)
        data = label['objects']
        img_name = image_id.split('.json')[0]
        img_name = img_name.split('\\')[-1]
        img_name = img_name.split('_det')[0]
        out_file = open('./dataset/yolodet/%s.txt' % (img_name), 'w', encoding='utf-8')
        for idx, obj in enumerate(data):
            cls = obj['category']
            # if cls not in classes:
            #     continue
            b1 = float(obj['box2d']['x1'])
            b3 = float(obj['box2d']['y1'])
            b2 = float(obj['box2d']['x2'])
            b4 = float(obj['box2d']['y2'])
            if cls == '5':
                cls = '0'
            elif cls == '6':
                cls ='1'

            w = 1920
            h = 1080
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            if b1 < 0:
                b1 = 0
            if b3 < 0:
                b3 = 0
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)

            out_file.write(str(cls) + " " +
                           " ".join([str(a) for a in bb]) + '\n')


# train_path = r'D:\code\test1\object detection\datasets\datasets\det_annotations\train'
# val_path = r'D:\code\test1\object detection\datasets\datasets\det_annotations\val'

val_path = '../dataset/json_det'
images = os.listdir(val_path)


for image in tqdm(list(images)):
    img_json = val_path + '\\'+image
    convert_annotation(img_json)

wd = getcwd()


