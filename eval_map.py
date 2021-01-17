import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import os
import cv2
from get_in4 import *
import random

def map(query, predict):
   mAp = 0.0
   num_correct = 0
   ap = 0.0
   max_query = len(predict)
   for i, pre in enumerate(predict):

        if query == int(pre):
            num_correct += 1

            ap += num_correct/(i+1)
   mAp = ap/float(max_query)
   return mAp


fe = FeatureExtractor()
features = []
img_paths = []
feature_paths = "./static/data_img/feature"
image_paths = "./static/data_img/image"
list_image = os.listdir(image_paths)
list_type_image = []
for i in list_image:
    list_type_image.append(i.split('_')[0])

querys = random.sample(list_image, 1000)
print('Loading feature & path image...')

for name_file in os.listdir(feature_paths):
    feature_path = feature_paths + '/' + name_file
    features.append(np.load(feature_path))
    img_paths.append(image_paths + '/' + name_file.split('.')[0]+".jpg")

features = np.array(features)


val_map = []
sum = 0
for j, qe in enumerate(querys):
    img = Image.open('static/data_img/image/' + qe)
    print(qe)
    query = fe.extract(img)
    dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:100]  # Top 30 results
    paths = []
    for i in range(len(ids)):
        paths.append(list_image[i].split('_')[0])
    print(paths)
    qe = int(qe.split('_')[0])
    a = map(qe, paths)
    val_map.append(a)
    sum += a
    print('map query', j, ': ', a)

print(val_map)
print(1.0*sum/len(val_map))


