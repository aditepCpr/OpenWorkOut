import numpy as np
import pickle
import matplotlib.pyplot as plt
from ReadData import CreateData as cd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


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


class Lori:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
        self.target_names = target_names
        self.name = 'LogisticRegression'
        self.nmesh = 200
        self.fileName = 'LogiReg'

    def lori(self):
        stored_lori = load_Data(self.fileName)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X, stored_lori, self.nmesh)
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


def load_Data(fileName):
    try:
        file_model = open(fileName + '.pkl', 'rb')
        model = pickle.load(file_model)
        file_model.close()
    except IOError as e:
        print(e)
    return model


def tuni(mz, name):
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


class DataModel:
    def __init__(self):
        self.squat = cd("dataSet/Squat")
        self.curl = cd("dataSet/Barbell Curl")
        self.pushup = cd('dataSet/Push Ups')
        self.dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
        self.deadlift = cd('dataSet/Deadlift')
        self.cam = cd('dataSet/cam')
        self.target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')

    def getSquat(self):
        return self.squat

    def getCurl(self):
        return self.curl

    def getPushup(self):
        return self.pushup

    def getDumbbellShoulderPress(self):
        return self.dumbbellShoulderPress

    def getDeadlift(self):
        return self.deadlift

    def getCam(self):
        return self.cam

    def getTargetNames(self):
        return self.target_names


class StackData:
    def __init__(self, path):
        self.path = path

    def stackData(self):
        idc = 0
        nxy, z = cd.allpath(self.path, idc)
        x = cd.xx(nxy)
        y = cd.yy(nxy)
        z = cd.cen_z(z)
        X = np.stack((x, y), axis=1)
        z = np.array(z)
        X = (X - X.mean(0)) / X.std(0)
        return X, z


if __name__ == '__main__':
    # path = [squat, curl, pushup, dumbbellShoulderPress, deadlift]
    try:
        dm = DataModel()
        cam = dm.getCam()
        target_names = dm.getTargetNames()
        path = [cam]
        data = StackData(path)
        X, z = data.stackData()
        # X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        print('DataSet OK...')
        # plt.scatter(X[:, 0], X[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
        # plt.show()
    except Exception as e:
        print(e)
    knn = Knn_(X, z, target_names)
    knn.knn()
    dt = DecisionTree(X, z, target_names)
    dt.decisionTree()
    randomforest = RandomForest(X, z, target_names)
    randomforest.randomforest()
    lori = Lori(X, z, target_names)
    lori.lori()
    # print(knn.mz_)
    # tuni(knn.mz_,knn.name)
    # tuni(knn.mz_,knn.name)
