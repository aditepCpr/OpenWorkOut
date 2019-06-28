import codecs, json
import numpy as np
from sklearn import datasets
import os
import matplotlib.pyplot as plt
def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    # print(path)
    X = []
    for kpose in path:
        try:
            file_path = (kpose)
            if (file_path != 'dataSet/__init__.py'):
                file = codecs.open(file_path, 'r', encoding='utf-8').read()
                b_new = json.loads(file)
                x = np.array(b_new)
                X.append(x)
        except IOError as e:
            print(e)
    xx = []
    yy = []
    for x in X:
        npx = np.array(x)
        x = npx[:, 0:1]
        y = npx[:, 1:2]
        for x1 in x :
            for x2 in x1:
                xx.append(x2)
        for y1 in y:
            for y2 in y1:
                yy.append(y2)

    plt.figure(figsize=[7, 7])  # กำหนดขนาดภาพให้เป็นจตุรัส
    plt.scatter(xx, yy)  # วาดแผนภาพการกระจาย
    plt.show()

train_classifier("dataSet")