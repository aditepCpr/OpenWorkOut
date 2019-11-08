import numpy as np
import pickle
import matplotlib.pyplot as plt
from StackData import DataModel
from StackData import StackData
from StackData import load_Data


class DecisionTree:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
        self.target_names = target_names
        self.name = ' Decision Tree '
        self.mz_ = self.mz_
        self.nmesh = 200
        self.fileName = 'DecisionTree'

    def decisionTree(self):
        stored_ditri = load_Data(self.fileName)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X, stored_ditri, self.nmesh)
        show_Data(self.X, self.z, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)


class RandomForest:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
        self.target_names = target_names
        self.name = 'random forest'
        self.mz_ = self.mz_
        self.nmesh = 200
        self.fileName = 'randomforest'

    def randomforest(self):
        stored_randomforest = load_Data(self.fileName)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X, stored_randomforest, self.nmesh)
        show_Data(self.X, self.z, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)


class Mlpc:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
        self.target_names = target_names
        self.name = 'Multi-layer Perceptron classifier'
        self.nmesh = 200
        self.fileName = 'MLPClassifier'

    def mlpc(self):
        stored_mlpc = load_Data(self.fileName)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X, stored_mlpc, self.nmesh)
        show_Data(self.X, self.z, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)


class Knn_:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
        self.target_names = target_names
        self.nmesh = 200
        self.name = 'k-nearest neighbor'
        self.mz_ = self.mz_
        self.fileName = 'Knn'

    def knn(self):
        stored_knn = load_Data(self.fileName)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X, stored_knn, self.nmesh)
        show_Data(self.X, self.z, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)


class SVC_:
    mz = []
    mz_ = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
        self.mz = self.mz
        self.mz_ = self.mz_
        self.target_names = target_names
        self.nmesh = 200
        self.name = 'support vector machine'
        self.fileName = 'Svc'

    def svc(self):
        stored_svc = load_Data(self.fileName)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X, stored_svc, self.nmesh)
        show_Data(self.X, self.z, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)


def predict_Data(X, model, nmesh):
    mz_ = model.predict(X)
    mx, my = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), nmesh),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), nmesh))
    mX = np.stack([mx.ravel(), my.ravel()], 1)
    mz = model.predict(mX).reshape(nmesh, nmesh)
    return mz_, mx, my, mX, mz


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
    X, z, target_names = data()

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
        # print(target_names[0])
        names = target_names[0]
    if x1 == max_h:
        # print(target_names[1])
        names = target_names[1]
    if x2 == max_h:
        # print(target_names[2])
        names = target_names[2]
    if x3 == max_h:
        # print(target_names[3])
        names = target_names[3]
    if x4 == max_h:
        # print(target_names[4])
        names = target_names[4]
    return names


def predict_knn():
    X, z, target_names = data()
    knn = Knn_(X, z, target_names)
    knn.knn()
    names = tuni(knn.mz_, knn.name)
    return names

def predict_DecisionTree():
    X, z, target_names = data()
    dt = DecisionTree(X, z, target_names)
    dt.decisionTree()
    names = tuni(dt.mz_, dt.name)
    return names

def predict_RandomForest():
    X, z, target_names = data()
    rt = RandomForest(X, z, target_names)
    rt.randomforest()
    names = tuni(rt.mz_, rt.name)
    return names

def predict_Mlpc():
    X, z, target_names = data()
    mlpc = Mlpc(X, z, target_names)
    mlpc.mlpc()
    names = tuni(mlpc.mz_, mlpc.name)
    return names


def predict_Svc():
    X, z, target_names = data()
    svc = SVC_(X, z, target_names)
    svc.svc()
    names = tuni(svc.mz_, svc.name)
    return names

def data():
    try:
        dm = DataModel()
        cam = dm.getCam()
        target_names = dm.getTargetNames()
        path = [cam]
        data = StackData(path)
        X, z = data.stackData_Predict()
        print('Predict DataSet OK...')
        # plt.scatter(X[:, 0], X[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
        # plt.show()
    except Exception as e:
        print(e)
    return X, z, target_names
