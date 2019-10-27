import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as logire
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier as Ditri
from sklearn.ensemble import RandomForestClassifier as Rafo
from StackData import DataModel
from StackData import StackData


class DecisionTree:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, X2, z2, target_names):
        self.X = X
        self.z = z
        self.X2 = X2
        self.z2 = z2
        self.mz = self.mz
        self.mx = self.mx
        self.my = self.my
        self.mX = self.mX
        self.target_names = target_names
        self.name = ' Decision Tree '
        self.mz_ = self.mz_
        self.nmesh = 200
        self.fileName = 'DecisionTree'

    def decisionTree(self):
        ditri = Ditri(max_depth=5)
        ditri.fit(self.X, self.z)

        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X2, ditri, self.nmesh)
        show_Data(self.X2, self.z2, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)
        dump_Data(self.fileName, ditri)


class RandomForest:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, X2, z2, target_names):
        self.X = X
        self.z = z
        self.X2 = X2
        self.z2 = z2
        self.mz = self.mz
        self.mx = self.mx
        self.my = self.my
        self.mX = self.mX
        self.target_names = target_names
        self.name = 'random forest'
        self.mz_ = self.mz_
        self.nmesh = 200
        self.fileName = 'randomforest'

    def randomforest(self):
        rafo = Rafo(n_estimators=100, max_depth=5)
        rafo = rafo.fit(self.X, self.z)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X2, rafo, self.nmesh)
        show_Data(self.X2, self.z2, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)
        dump_Data(self.fileName, rafo)


class Lori:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, X2, z2, target_names):
        self.X = X
        self.z = z
        self.X2 = X2
        self.z2 = z2
        self.mz = self.mz
        self.mx = self.mx
        self.my = self.my
        self.mX = self.mX
        self.mz_ = self.mz_
        self.target_names = target_names
        self.name = 'LogisticRegression'
        self.nmesh = 200
        self.fileName = 'LogiReg'

    def lori(self):
        lori = logire()
        lori.fit(self.X, self.z)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X2, lori, self.nmesh)
        show_Data(self.X2, self.z2, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)
        dump_Data(self.fileName, lori)


class Knn_:
    mz = []
    mz_ = []

    def __init__(self, X, z, X2, z2, target_names):
        self.X = X
        self.z = z
        self.X2 = X2
        self.z2 = z2
        self.mz = self.mz
        self.target_names = target_names
        self.nmesh = 200
        self.name = 'k-nearest neighbor'
        self.mz_ = self.mz_
        self.fileName = 'Knn'

    def knn(self):
        knn = Knn(n_neighbors=1, p=1, algorithm='kd_tree', n_jobs=-1)
        knn = knn.fit(self.X, self.z)
        parameters = {'n_neighbors': range(1, 11)}
        knn_best = GridSearchCV(knn, parameters, cv=5)
        knn_best.fit(self.X, self.z)
        knn_best.best_estimator_

        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X2, knn_best, self.nmesh)
        show_Data(self.X2, self.z2, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)
        dump_Data(self.fileName, knn_best)


def dump_Data(fileName, model):
    try:
        f = open(fileName + '.pkl', 'wb')
        pickle.dump(model, f)
        f.close()
        print('Dump_file OK...')
    except IOError as e:
        print(e)


def predict_Data(X, model, nmesh):
    mz_ = model.predict(X)
    mx, my = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), nmesh),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), nmesh))
    mX = np.stack([mx.ravel(), my.ravel()], 1)
    mz = model.predict(mX).reshape(nmesh, nmesh)
    return mz_, mx, my, mX, mz


def show_Data(X, z, mx, my, mz, name, target_names, mz_):
    print('#' * 25 + name + '#' * 25)
    print(classification_report(z, mz_, target_names=target_names))
    print('accuracy_score = ', accuracy_score(z, mz_))
    print('#' * 60)
    plt.figure().gca(aspect=1, xlim=[mx.min(), mx.max()], ylim=[my.min(), my.max()])
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c=z, edgecolor='k', cmap='rainbow')
    plt.title(name)
    plt.contourf(mx, my, mz, alpha=0.4, cmap='rainbow', zorder=0)
    plt.show()

if __name__ == '__main__':
    try:

        dataModel = DataModel()
        squat = dataModel.getSquat()
        curl = dataModel.getCurl()
        pushup = dataModel.getPushup()
        dumbbellShoulderPress = dataModel.getDumbbellShoulderPress()
        deadlift = dataModel.getDeadlift()
        cam = dataModel.getCam()
        target_names = dataModel.getTargetNames()

        path = [squat, curl, pushup, dumbbellShoulderPress, deadlift]
        sd = StackData(path)
        X_train, X_test, z_train, z_test = sd.stackData_Train()
        print('DataSet OK...')
        # plt.scatter(X[:, 0], X[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
        # plt.show()
    except Exception as e:
        print(e)

    knn = Knn_(X_train, z_train, X_test, z_test, target_names)
    knn.knn()
    dt = DecisionTree(X_train, z_train, X_test, z_test, target_names)
    dt.decisionTree()
    randomforest = RandomForest(X_train, z_train, X_test, z_test, target_names)
    randomforest.randomforest()
    lori = Lori(X_train, z_train, X_test, z_test, target_names)
    lori.lori()
    # # tuni(d_tree.mz, 'decision tree')
