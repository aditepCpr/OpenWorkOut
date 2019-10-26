import numpy as np
import pickle
import matplotlib.pyplot as plt
from ReadData import CreateData as cd
from sklearn.linear_model import LogisticRegression as logire
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier as Ditri


class DecisionTree:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
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
        stored_lori = load_Data(self.fileName)
        self.mz_, self.mx, self.my, self.mX, self.mz = predict_Data(self.X, stored_lori, self.nmesh)
        show_Data(self.X, self.z, self.mx, self.my, self.mz, self.name, self.target_names, self.mz_)


class Knn_:
    mz = []
    mz_ = []

    def __init__(self, X, z, target_names):
        self.X = X
        self.z = z
        self.mz = self.mz
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
    print('#' * 25 + name + '#' * 25)
    print(classification_report(z, mz_, target_names=target_names))
    print('accuracy_score = ', accuracy_score(z, mz_))
    print('#' * 60)
    plt.figure().gca(aspect=1, xlim=[mx.min(), mx.max()], ylim=[my.min(), my.max()])
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c=z, edgecolor='k', cmap='rainbow')
    plt.title(name)
    plt.contourf(mx, my, mz, alpha=0.4, cmap='rainbow', zorder=0)
    plt.show()

def load_Data(fileName):
    try:
        file_model = open(fileName+'.pkl', 'rb')
        model = pickle.load(file_model)
        file_model.close()
    except IOError as e:
        print(e)
    return model



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
    dt = DecisionTree( X_test, z_test, target_names)
    dt.decisionTree()
    randomforest = RandomForest(X_test, z_test, target_names)
    randomforest.randomforest()
    lori = Lori(X_test,z_test,target_names)
    lori.lori()
    # tuni(d_tree.mz, 'decision tree')
