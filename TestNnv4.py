import numpy as np
import pickle
import matplotlib.pyplot as plt
from ReadData import CreateData as cd
from sklearn.linear_model import LogisticRegression as logire
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.model_selection import GridSearchCV


class D_tree:
    mz = []
    mx = []
    my = []
    mX = []

    def __init__(self, X, z, target_names):

        self.X = X
        self.z = z
        self.mz = self.mz
        self.mx = self.mx
        self.my = self.my
        self.mX = self.mX
        self.target_names = target_names

    def d_tree(self):
        try:
            file_model = open('d_t.pkl', 'rb')
            stored_d_t = pickle.load(file_model)
            file_model.close()
        except IOError as e:
            print(e)
        self.mz = stored_d_t.predict(self.X)
        print(classification_report(self.z, self.mz, target_names=self.target_names))
        print(accuracy_score(self.z, self.mz))
        nmesh = 2000
        self.mx, self.my = np.meshgrid(np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), nmesh),
                                       np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), nmesh))
        self.mX = np.stack([self.mx.ravel(), self.my.ravel()], 1)
        self.mz = stored_d_t.predict(self.mX).reshape(nmesh, nmesh)
        plottassimo(self.X, self.z, self.mx, self.my, self.mz, 'Decision tree')


class Lori:
    mz = []

    def __init__(self, X, z, X2, z2, target_names):
        self.X = X
        self.z = z
        self.X2 = X2
        self.z2 = z2
        self.target_names = target_names

    def lori(self):
        print(self.X)
        lori = logire()
        lori.fit(self.X, self.z)
        self.mz = lori.predict(self.X2)
        print(classification_report(self.z2, self.mz, target_names=self.target_names))
        print(accuracy_score(self.z2, self.mz))

        nmesh = 2000
        self.mx, self.my = np.meshgrid(np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), nmesh),
                                       np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), nmesh))
        self.mX = np.stack([self.mx.ravel(), self.my.ravel()], 1)
        self.mz = lori.predict(self.mX).reshape(nmesh, nmesh)
        plottassimo(self.X, self.z, self.mx, self.my, self.mz, 'LogisticRegression')


class Knn_:
    print('knn')
    mz = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
        self.mz = self.mz
        self.target_names = target_names
        self.nmesh = 200

    def knn(self):
        try:
            file_model = open('knn.pkl', 'rb')
            stored_knn = pickle.load(file_model)
            file_model.close()
        except IOError as e:
            print(e)

        self.mz = stored_knn.predict(self.X)
        print('#' * 25 + '    Knn    ' + '#' * 25)
        print(classification_report(self.z, self.mz, target_names=self.target_names))
        print('accuracy_score = ', accuracy_score(self.z, self.mz))
        print('#' * 60)
        self.mx, self.my = np.meshgrid(np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), self.nmesh),
                                       np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), self.nmesh))
        self.mX = np.stack([self.mx.ravel(), self.my.ravel()], 1)
        self.mz = stored_knn.predict(self.mX).reshape(self.nmesh, self.nmesh)
        plottassimo(self.X, self.z, self.mx, self.my, self.mz, 'Decision tree')


def plottassimo(X, z, mx, my, mz, name):
    plt.figure().gca(aspect=1, xlim=[mx.min(), mx.max()], ylim=[my.min(), my.max()])
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c=z, edgecolor='k', cmap='rainbow')
    plt.title(name)
    plt.contourf(mx, my, mz, alpha=0.4, cmap='rainbow', zorder=0)
    plt.show()


def tuni(mz, name):
    print(name)
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    for h in mz:

        for h1 in h:
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


squat = cd("dataSet/Squat")
curl = cd("dataSet/Barbell Curl")
pushup = cd('dataSet/Push Ups')
dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
deadlift = cd('dataSet/Deadlift')
cam = cd('dataSet/cam')
target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')

path = [squat, curl, pushup, dumbbellShoulderPress, deadlift]
idc = 0
nxy, z = cd.allpath(path, idc)
x = cd.xx(nxy)
y = cd.yy(nxy)
z = cd.cen_z(z)
X = np.stack((x, y), axis=1)
z = np.array(z)
X = (X - X.mean(0)) / X.std(0)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
print('Showdata OK...')
# plt.scatter(X[:, 0], X[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
# plt.show()

if __name__ == '__main__':
    print(__name__)
    knn = Knn_(X_test, z_test, target_names)
    knn.knn()
    # d_tree = D_tree(X_test, z_test, target_names)
    # d_tree.d_tree()
    # lori = Lori(X_train,z_train,X_test,z_test,target_names)
    # lori.lori()
    # tuni(d_tree.mz, 'decision tree')
