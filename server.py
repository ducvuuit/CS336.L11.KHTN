import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from get_in4 import *
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import pandas as pd
import os
import cv2 as cv
import _pickle as cPickle
from ransac import ransac
import time


data = pd.read_csv("static/data_img/data.csv")
name_img = data['name_img']
link = data['link']
price = data['price']
app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
feature_paths = "./static/data_img/feature"
image_paths = "./static/data_img/image"
des_paths = "./static/data_img/des"
kp_paths = "static/data_img/keypoint"
print('Loading feature & path image...')

for name_file in os.listdir(feature_paths):
    feature_path = feature_paths + '/' + name_file
    features.append(np.load(feature_path))
    img_paths.append(image_paths + '/' + name_file.split('.')[0]+".jpg")

features = np.array(features)
print('Loaded')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        start = time.time()
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        ran = []
        ids_new = []

        # get des, kp from query img
        img1 = cv.imread(uploaded_img_path, 0)  # queryImage
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)

        for id in ids:
            ran.append(ransac(img1, des1, kp1, img_paths[id]))
        ran = np.argsort(ran)[::-1]
        for i in ran:
            ids_new.append(ids[i])
        re = len(ids)
        # #
        scores = []
        for id in ids:
            name_product, link, price = get_in4_from_positon(id)
            scores.append([dists[id], img_paths[id], name_product, link, price])

        end = time.time()
        run_time = end - start
        return render_template('home-03.html',
                               query_path=uploaded_img_path,
                               scores=scores,
                               re=re,
                               run_time=run_time
                               )
        # name_products=name_products,
        # links=links,
        # prices=prices)
    else:
        return render_template('home-03.html')


if __name__=="__main__":
    app.config['ENV'] = 'development'
    app.config['DEBUG'] = True
    app.config['TESTING'] = True
    app.run("127.0.0.1")