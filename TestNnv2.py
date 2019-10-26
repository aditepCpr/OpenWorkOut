from ReadData import CreateData as cd
# from ReadData_2 import CreateData2 as cd
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as Knn
from unagi import Affin, Softmax_entropy, Sigmoid_entropy, ha_1h
from unagi import Sigmoid, Relu, Lrelu, Prelu, Elu, Selu, Tanh, Softsign, Softplus
from sklearn.ensemble import RandomForestClassifier as Rafo
from sklearn.preprocessing import StandardScaler as Sta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


class Sgd:
    def __init__(self, param, eta=0.01):
        self.param = param
        self.eta = eta

    def __call__(self):
        for p in self.param:
            p.kha -= self.eta * p.g
            p.g = 0


class Mmtsgd:
    def __init__(self, param, eta=0.01, mmt=0.9):
        self.param = param
        self.eta = eta
        self.mmt = mmt
        self.d = [0] * len(param)

    def __call__(self):
        for i, p in enumerate(self.param):
            self.d[i] = self.mmt * self.d[i] - self.eta * p.g
            p.kha += self.d[i]
            p.g = 0


class Nag:
    def __init__(self, param, eta=0.01, mmt=0.9):
        self.param = param
        self.eta = eta
        self.mmt = mmt
        self.d = [0] * len(param)
        self.g0 = np.nan

    def __call__(self):
        if (self.g0 is np.nan):
            self.g0 = [p.g for p in self.param]
        for i, p in enumerate(self.param):
            self.d[i] = self.mmt * self.d[i] - self.eta * (p.g + self.mmt * (p.g - self.g0[i]))
            self.g0[i] = p.g
            p.kha += self.d[i]
            p.g = 0


class Adagrad:
    def __init__(self, param, eta=0.01):
        self.param = param
        self.eta = eta
        self.G = [1e-7] * len(param)

    def __call__(self):
        for i, p in enumerate(self.param):
            self.G[i] += p.g ** 2
            p.kha += -self.eta * p.g / np.sqrt(self.G[i])
            p.g = 0


class Adadelta:
    def __init__(self, param, eta=0.01, rho=0.95):
        self.param = param
        self.eta = eta
        self.rho = rho
        self.G = [1e-7] * len(param)

    def __call__(self):
        for i, p in enumerate(self.param):
            self.G[i] = self.rho * self.G[i] + (1 - self.rho) * p.g ** 2
            p.kha += -self.eta * p.g / np.sqrt(self.G[i])
            p.g = 0


class Adam:
    def __init__(self, param, eta=0.001, beta1=0.9, beta2=0.999):
        self.param = param
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        n = len(param)
        self.m = [0] * n
        self.v = [1e-7] * n
        self.t = 1

    def __call__(self):
        for i, p in enumerate(self.param):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.g ** 2
            p.kha += -self.eta * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t) * self.m[i] / np.sqrt(
                self.v[i])
            self.t += 1
            p.g = 0


class Prasat:
    def __init__(self, m, sigma=1, eta=0.1, kratun='relu', opt='adam'):
        self.m = m
        self.chan = []
        for i in range(len(m) - 1):
            self.chan.append(Affin(m[i], m[i + 1], sigma))
            if (i < len(m) - 2):
                if (kratun == 'relu'):
                    self.chan.append(Relu())
                else:
                    self.chan.append(Sigmoid())
        self.chan.append(Softmax_entropy())
        opt = {'adam': Adam, 'adagrad': Adagrad,
               'adadelta': Adadelta, 'mmtsgd': Mmtsgd,
               'sgd': Sgd, 'nag': Nag}[opt]
        self.opt = opt(self.param(), eta=eta)

    def rianru(self, X, z, n_thamsam):
        Z = ha_1h(z, self.m[-1])
        for i in range(n_thamsam):
            entropy = self.ha_entropy(X, Z)
            entropy.phraeyon()
            self.opt()

    def ha_entropy(self, X, Z):
        for c in self.chan[:-1]:
            X = c(X)
        return self.chan[-1](X, Z)

    def param(self):
        p = []
        for c in self.chan:
            if (hasattr(c, 'param')):
                p.extend(c.param)
        return p

    def thamnai(self, X):
        for c in self.chan[:-1]:
            X = c(X)
        return X.kha.argmax(1)


squat = cd("dataSet/Squat")
curl = cd("dataSet/Barbell Curl")
pushup = cd('dataSet/Push Ups')
dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
deadlift = cd('dataSet/Deadlift')
cam = cd('dataSet/cam')
# target_names = np.array(['curl','pushup', 'squat', 'deadlift'], dtype='<U10')
target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')
# target_names = np.array(['dumbbellShoulderPress', 'squat'], dtype='<U10')

path = [squat, curl, pushup, dumbbellShoulderPress, deadlift]
# path = [squat,curl]
idc = 0
nxy, z = cd.allpath(path, idc)
x = cd.xx(nxy)
y = cd.yy(nxy)
z = cd.cen_z(z)
X = np.stack((x, y), axis=1)
z = np.array(z)
xy_sta = (X - X.mean(0)) / X.std(0)
X = xy_sta
print('Showdata OK...')
plt.scatter(xy_sta[:, 0], xy_sta[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
plt.show()
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

# sta = Sta()
# sta.fit(X)
# xy_sta = sta.transform(X)
# plt.scatter(xy_sta[:, 0], xy_sta[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
# plt.show()

# path2 = [dumbbellShoulderPress]
# idc2 = 0
# nxy2, z2 = cd.allpath(path2, idc2)
# x2 = cd.xx(nxy2)
# y2 = cd.yy(nxy2)
# z2 = cd.cen_z(z2)
# X2 = np.stack((x2, y2), axis=1)
# xy_sta2 = (X2 - X2.mean(0)) / X2.std(0)
# X2 = xy_sta2
# plt.scatter(xy_sta2[:, 0], xy_sta2[:, 1], 50, c=z2, edgecolor='k', cmap='rainbow')
# plt.show()

import seaborn as sns


#####################################################################################
class Knn_v1:
    mz = []
    mx = []
    my = []
    mX = []

    def __init__(self, X, z, X2, z2,target_names):
        self.X = X
        self.z = z
        self.X2 = X2
        self.z2 = z2
        self.mz = self.mz
        self.mx = self.mx
        self.my = self.my
        self.mX = self.mX
        self.nmesh = 200
        self.target_names = target_names

    def Knn_v1(self):

        knn = Knn(n_neighbors=1, p=1, algorithm='kd_tree', n_jobs=-1)
        knn = knn.fit(self.X, self.z)

        ### KNN ธรรมดา
        # self.mz = knn.predict(self.X2)
        # print(classification_report(self.z2, self.mz))
        # self.mx, self.my = np.meshgrid(np.linspace(self.X2[:, 0].min(), self.X2[:, 0].max(), self.nmesh),
        #                                np.linspace(self.X2[:, 1].min(), self.X2[:, 1].max(), self.nmesh))
        # self.mX = np.stack([self.mx.ravel(), self.my.ravel()], 1)
        #
        # self.mz = knn.predict(self.mX).reshape(self.nmesh, self.nmesh)
        #
        # plt.gca(xlim=[self.X2[:, 0].min(), self.X2[:, 0].max()], ylim=[self.X2[:, 1].min(), self.X2[:, 1].max()],
        #         aspect=1)
        # plt.contourf(self.mx, self.my, self.mz, alpha=0.1, cmap='rainbow')
        # plt.contour(self.mx, self.my, self.mz, colors='#222222')
        # plt.title("KNN")
        # plt.scatter(self.X2[:, 0], self.X2[:, 1], c=self.z2, edgecolor='k', cmap='rainbow')
        # plt.show()

        #### knn แบบ การคำนวณความน่าจะเป็น
        # self.mz = knn.predict_proba(self.mX)[:, len(path2)].reshape(self.nmesh, self.nmesh)
        # plt.gca(xlim=[self.X2[:, 0].min(), self.X2[:, 0].max()], ylim=[self.X2[:, 1].min(), self.X2[:, 1].max()],
        #         aspect=1)
        # plt.contourf(self.mx, self.my, self.mz, alpha=0.1, cmap='rainbow')
        # plt.contour(self.mx, self.my, self.mz, colors='#222222')
        # plt.scatter(self.X2[:, 0], self.X2[:, 1], c=self.z2, edgecolor='k', cmap='rainbow')
        # plt.show()

        #### knn แบบปรับ parameter
        parameters = {'n_neighbors': range(1, 11)}
        knn_best = GridSearchCV(knn, parameters, cv=5)
        knn_best.fit(self.X, self.z)
        knn_best.best_estimator_
        self.mz = knn_best.predict(self.X2)
        print(classification_report(self.z2, self.mz,target_names=self.target_names))
        self.mx, self.my = np.meshgrid(np.linspace(self.X2[:, 0].min(), self.X2[:, 0].max(), self.nmesh),
                                       np.linspace(self.X2[:, 1].min(), self.X2[:, 1].max(), self.nmesh))
        self.mX = np.stack([self.mx.ravel(), self.my.ravel()], 1)

        self.mz = knn_best.predict(self.mX).reshape(self.nmesh, self.nmesh)

        plt.gca(xlim=[self.X2[:, 0].min(), self.X2[:, 0].max()], ylim=[self.X2[:, 1].min(), self.X2[:, 1].max()],
                aspect=1)
        plt.contourf(self.mx, self.my, self.mz, alpha=0.1, cmap='rainbow')
        plt.contour(self.mx, self.my, self.mz, colors='#222222')
        plt.title("KNN_best")
        plt.scatter(self.X2[:, 0], self.X2[:, 1], c=self.z2, edgecolor='k', cmap='rainbow')
        plt.show()

        import pickle
        f = open('knn.pkl', 'wb')
        pickle.dump(knn_best, f)
        f.close()


###################################################################################
class Nn_v1:
    mz = []

    def __init__(self, X, z, X2, z2):
        self.X = X
        self.z = z
        self.X2 = X2
        self.z2 = z2
        self.mz = self.mz

    def nn_v1(self):
        prasat = Prasat(m=[2, 100, 100, 100, 100, len(path)], eta=0.005)
        prasat.rianru(self.X, self.z, n_thamsam=1000)
        mx, my = np.meshgrid(np.linspace(self.X2[:, 0].min(), self.X2[:, 0].max(), 200),
                             np.linspace(self.X2[:, 1].min(), self.X2[:, 1].max(), 200))
        mX = np.array([mx.ravel(), my.ravel()]).T
        self.mz = prasat.thamnai(mX).reshape(200, -1)
        plt.gca(aspect=1, xlim=(self.X2[:, 0].min(), self.X2[:, 0].max()),
                ylim=(self.X2[:, 1].min(), self.X2[:, 1].max()))
        plt.contourf(mx, my, self.mz, cmap='rainbow', alpha=0.2)
        plt.scatter(self.X2[:, 0], self.X2[:, 1], 50, c=self.z2, edgecolor='k', cmap='rainbow')
        plt.show()

        plt.figure(figsize=[6, 10])
        ax1 = plt.subplot(211, xticks=[])
        ax1.set_title(u'entropy')
        ax2 = plt.subplot(212)
        ax2.set_title(u'Score')
        for Opt in [Sgd, Mmtsgd, Nag, Adagrad, Adadelta, Adam]:
            chan = [Affin(2, 60, 1), Sigmoid(), Affin(60, 1, 1), Sigmoid_entropy()]
            opt = Opt(chan[0].param + chan[2].param, eta=0.02)
            lis_entropy = []
            lis_khanaen = []
            for i in range(200):
                X_ = self.X
                for c in chan[:-1]:
                    X_ = c(X_)
                lis_khanaen.append(((X_.kha.ravel() > 0) == z).mean())
                entropy = chan[-1](X_, z)
                lis_entropy.append(entropy.kha)
                entropy.phraeyon()
                opt()
            si = np.random.random(3)
            ax1.plot(lis_entropy, color=si)
            ax2.plot(lis_khanaen, color=si)
        plt.legend(['SGD', 'Momentum', 'NAG', 'AdaGrad', 'AdaDelta', 'Adam'], ncol=2)
        plt.tight_layout()
        plt.show()

        # f = open('nn.pkl', 'wb')
        # pickle.dump(prasat, f)
        # f.close()


###################################################################################
def tuni(mz, name):
    print(name)
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    for h in mz:
        # print(h)
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


###################################################################################
class D_tree:
    mz = []
    mx = []
    my = []
    mX = []

    def __init__(self, X, z, X2, z2):
        self.X = X
        self.z = z
        self.X2 = X2
        self.z2 = z2
        self.mz = self.mz
        self.mx = self.mx
        self.my = self.my
        self.mX = self.mX
        self.khanaen_fuek = []
        self.khanaen_truat = []

    def d_tree(self):
        nmesh = 2000
        for i in range(1, 21):
            rafo = Rafo(n_estimators=100, max_depth=i)
            rafo = rafo.fit(self.X, self.z)
            self.khanaen_fuek.append(rafo.score(self.X, self.z))
            self.khanaen_truat.append(rafo.score(self.X2, self.z2))
        self.mz = rafo.predict(self.X2)
        print(classification_report(self.z2, self.mz, target_names=target_names))

        self.mx, self.my = np.meshgrid(np.linspace(self.X2[:, 0].min(), self.X2[:, 0].max(), nmesh),
                                       np.linspace(self.X2[:, 1].min(), self.X2[:, 1].max(), nmesh))
        self.mX = np.stack([self.mx.ravel(), self.my.ravel()], 1)
        self.mz = rafo.predict(self.mX).reshape(nmesh, nmesh)
        self.plottassimo(self.X2, self.z2, self.mx, self.my, self.mz)

        f = open('d_t.pkl', 'wb')
        pickle.dump(rafo, f)
        f.close()

    def plottassimo(self, X, z, mx, my, mz):
        plt.figure().gca(aspect=1, xlim=[mx.min(), mx.max()], ylim=[my.min(), my.max()])
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c=z, edgecolor='k', cmap='rainbow')
        plt.title("decision tree ")
        plt.contourf(mx, my, mz, alpha=0.4, cmap='rainbow', zorder=0)
        plt.show()

        plt.plot(range(1, 21), self.khanaen_fuek, '#771133')
        plt.plot(range(1, 21), self.khanaen_truat, '#117733')
        plt.legend([u'Training', u'Test'])
        plt.show()


###################################################################################
if __name__ == '__main__':
    print(__name__)
    knn_v1 = Knn_v1(X_train, z_train, X_test, z_test,target_names)
    knn_v1.Knn_v1()
    # nn_v1 = Nn_v1(X, z, X2, z2)
    # nn_v1.nn_v1()
    # d_tree = D_tree(X_train, z_train, X_test, z_test)
    # d_tree.d_tree()

    # # tuni(knn_v1.mz, 'Knn')
    # # tuni(nn_v1.mz, 'Nn')
    # tuni(d_tree.mz, 'd_t')
