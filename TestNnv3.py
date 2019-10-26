
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
        self.kratun = kratun

        print('Start...')
        print('kratun', self.kratun)
        print('eta', eta)
        print('opt', opt)
        for i in range(len(m) - 1):
            self.chan.append(Affin(m[i], m[i + 1], sigma))
            if (i < len(m) - 2):
                if (self.kratun == 'relu'):
                    self.chan.append(Relu())
                if (self.kratun == 'Lrelu'):
                    self.chan.append(Lrelu())
                if (self.kratun == 'Prelu'):
                    self.chan.append(Prelu(m))
                if (self.kratun == 'Elu'):
                    self.chan.append(Elu())
                if (self.kratun == 'Selu'):
                    self.chan.append(Selu())
                if (self.kratun == 'Tanh'):
                    self.chan.append(Tanh())
                if (self.kratun == 'Softsign'):
                    self.chan.append(Softsign())
                if (self.kratun == 'Softplus'):
                    self.chan.append(Softplus())
                if (self.kratun == 'Sigmoid'):
                    self.chan.append(Sigmoid())
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
# target_names = np.array(['squat', 'pushup', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')

path = [squat, curl, pushup, dumbbellShoulderPress, deadlift]
# path = [cam]
idc = 0
nxy, z = cd.allpath(path, idc)
x = cd.xx(nxy)
y = cd.yy(nxy)
z = cd.cen_z(z)
X = np.stack((x, y), axis=1)
X2 = np.stack((x, y), axis=1)
z = np.array(z)
z2 = np.array(z)
# c = cd.c(x,y,z)
print('data OK...')

plt.scatter(X[:, 0], X[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
plt.show()

# path2 = [cam]
# idc2 = 0
# nxy2, z2 = cd.allpath(path2, idc2)
# x2 = cd.xx(nxy2)
# y2 = cd.yy(nxy2)
# z2 = cd.cen_z(z2)
# X2 = np.stack((x2, y2), axis=1)
# plt.scatter(X2[:, 0], X2[:, 1], 50, c=z2, edgecolor='k', cmap='rainbow')
# plt.show()

# chue = [u'ReLU', u'LReLU(0.1)', u'PReLU(0.25)', u'ELU(1)', u'SELU', u'softplus', u'sigmoid', u'tanh', u'softsign']
# plt.figure(figsize=[7, 5])
# for i, f in enumerate([Relu(), Lrelu(0.1), Prelu(1, 0.25), Elu(1), Selu(), Softplus(), Sigmoid(), Tanh(), Softsign()]):
#     a = np.linspace(-4, 4, 201)
#     h = f.pai(a)
#     print(i)
#     plt.subplot(331 + i, aspect=1, xlim=[-4, 4], xticks=range(-5, 6), yticks=range(-5, 6))
#     plt.title(chue[i])
#     plt.plot(a, h, color=np.random.random(3), lw=2)
#     plt.grid(ls='--')
# plt.tight_layout()
# plt.show()

kratun = ['Lrelu', 'relu', 'Elu','Selu','Tanh','Softsign']
plt.figure(figsize=[9, 8])
i = 0
for N_kratun in kratun:
    prasat = Prasat(m=[2, 100, 100, 100, len(path)], eta=0.005, kratun=N_kratun)
    prasat.rianru(X, z, n_thamsam=1000)
    mx, my = np.meshgrid(np.linspace(X2[:, 0].min(), X2[:, 0].max(), 200),
                         np.linspace(X2[:, 1].min(), X2[:, 1].max(), 200))
    mX = np.array([mx.ravel(), my.ravel()]).T
    mz = prasat.thamnai(mX).reshape(200, -1)

    plt.gca(aspect=1, xlim=(X2[:, 0].min(), X2[:, 0].max()),
            ylim=(X2[:, 1].min(), X2[:, 1].max()))

    plt.subplot(321 + i, aspect=1, xlim=[mx.min(), mx.max()], ylim=[my.min(), my.max()])
    plt.title(N_kratun)
    plt.scatter(X2[:, 0], X2[:, 1], 50, c=z2, edgecolor='k', cmap='rainbow')
    plt.contourf(mx, my, mz, cmap='rainbow', alpha=0.2)
    i += 1
plt.tight_layout()
plt.show()


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


tuni(mz, 'Nn')
