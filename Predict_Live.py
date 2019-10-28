import numpy as np
import pickle
import matplotlib.pyplot as plt
from StackData import DataModel
from StackData import StackData
from StackData import load_Data


class Predict_Live:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, target_names):
        self.X = X
        self.z = np.zeros(len(self.X))
        self.target_names = target_names
        self.nmesh = 200
        self.name = 'k-nearest neighbor'
        self.mz_ = self.mz_
        self.fileName = 'Knn'

    def predictLive(self):
        stored_knn = load_Data(fileName=self.fileName)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X, stored_knn)

def predict_Data(X, model):
    mz_ = model.predict(X)
    return mz_

def show_Data(X, z, mx, my, mz, name, target_names, mz_):
    # print('#' * 25 + name + '#' * 25)
    # print(classification_report(z, mz_, target_names=target_names))
    # print('accuracy_score = ', accuracy_score(z, mz_))
    # print('#' * 60)
    plt.figure().gca(aspect=1, xlim=[mx.min(), mx.max()], ylim=[my.min(), my.max()])
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c=z, edgecolor='k', cmap='rainbow')
    plt.title(name)
    plt.contourf(mx, my, mz, alpha=0.4, cmap='rainbow', zorder=0)
    plt.show()

def tuni(mz, name):
    dm = DataModel()
    target_names = dm.getTargetNames()

    print(name)
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    for h1 in mz:
        if h1 == 0:
            x0 += 1
        elif h1 == 1:
            x1 += 1
        elif h1 == 2:
            x2 += 1
        elif h1 == 3:
            x3 += 1
        elif h1 == 4:
            x4 += 1

    max_h = max(x0, x1, x2, x3, x4)
    print(name, target_names[0], 'x0', x0)
    print(name, target_names[1], 'x1', x1)
    print(name, target_names[2], 'x2', x2)
    print(name, target_names[3], 'x3', x3)
    print(name, target_names[4], 'x4', x4)
    if x0 == max_h:
        print(target_names[0])
    if x1 == max_h:
        print(target_names[1])
    if x2 == max_h:
        print(target_names[2])
    if x3 == max_h:
        print(target_names[3])
    if x4 == max_h:
        print(target_names[4])

from body.KeyPoints import KeyPoints
def Live(kp):
    dm = DataModel()
    target_names = dm.getTargetNames()
    X_n = KeyPoints.getAllKeypoints(kp)
    pl = Predict_Live(X_n,target_names)
    pl.predictLive()
    tuni(pl.mz_,'KNN')
