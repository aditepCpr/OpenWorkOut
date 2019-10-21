import numpy as np
import matplotlib.pyplot as plt
from unagi import Affin, Sigmoid, Sigmoid_entropy, Adam, Relu
from createData import CreateData as cd
import matplotlib.pyplot as plt
from unagi import Affin,Sigmoid,Relu,Softmax_entropy,Sigmoid_entropy,ha_1h

class Sgd:
    def __init__(self,param,eta=0.01):
        self.param = param
        self.eta = eta

    def __call__(self):
        for p in self.param:
            p.kha -= self.eta*p.g
            p.g = 0

class Mmtsgd:
    def __init__(self,param,eta=0.01,mmt=0.9):
        self.param = param
        self.eta = eta
        self.mmt = mmt
        self.d = [0]*len(param)

    def __call__(self):
        for i,p in enumerate(self.param):
            self.d[i] = self.mmt*self.d[i]-self.eta*p.g
            p.kha += self.d[i]
            p.g = 0

class Nag:
    def __init__(self,param,eta=0.01,mmt=0.9):
        self.param = param
        self.eta = eta
        self.mmt = mmt
        self.d = [0]*len(param)
        self.g0 = np.nan

    def __call__(self):
        if(self.g0 is np.nan):
            self.g0 = [p.g for p in self.param]
        for i,p in enumerate(self.param):
            self.d[i] = self.mmt*self.d[i]-self.eta*(p.g+self.mmt*(p.g-self.g0[i]))
            self.g0[i] = p.g
            p.kha += self.d[i]
            p.g = 0

class Adagrad:
    def __init__(self,param,eta=0.01):
        self.param = param
        self.eta = eta
        self.G = [1e-7]*len(param)

    def __call__(self):
        for i,p in enumerate(self.param):
            self.G[i] += p.g**2
            p.kha += -self.eta*p.g/np.sqrt(self.G[i])
            p.g = 0

class Adadelta:
    def __init__(self,param,eta=0.01,rho=0.95):
        self.param = param
        self.eta = eta
        self.rho = rho
        self.G = [1e-7]*len(param)

    def __call__(self):
        for i,p in enumerate(self.param):
            self.G[i] = self.rho*self.G[i]+(1-self.rho)*p.g**2
            p.kha += -self.eta*p.g/np.sqrt(self.G[i])
            p.g = 0

class Adam:
    def __init__(self,param,eta=0.001,beta1=0.9,beta2=0.999):
        self.param = param
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        n = len(param)
        self.m = [0]*n
        self.v = [1e-7]*n
        self.t = 1

    def __call__(self):
        for i,p in enumerate(self.param):
            self.m[i] = self.beta1*self.m[i]+(1-self.beta1)*p.g
            self.v[i] = self.beta2*self.v[i]+(1-self.beta2)*p.g**2
            p.kha += -self.eta*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)*self.m[i]/np.sqrt(self.v[i])
            self.t += 1
            p.g = 0

class Prasat:
    def __init__(self, m, eta, kratun='relu'):
        m.append(1)
        self.m = m
        self.chan = []
        for i in range(len(m) - 1):
            self.chan.append(Affin(m[i], m[i + 1], np.sqrt(2. / m[i])))
            if (i < len(m) - 2):
                if (kratun == 'relu'):
                    self.chan.append(Relu())
                else:
                    self.chan.append(Sigmoid())
        self.chan.append(Sigmoid_entropy())
        self.opt = Adam(self.param(), eta=eta)

    def rianru(self, X, z, n_thamsam, n_batch=50):
        n = len(z)
        self.entropy = []
        self.khanaen = []
        for o in range(n_thamsam):
            lueak = np.random.permutation(n)
            for i in range(0, n, n_batch):
                Xb = X[lueak[i:i + n_batch]]
                zb = z[lueak[i:i + n_batch]]
                entropy = self.ha_entropy(Xb, zb)
                entropy.phraeyon()
                self.opt()
            entropy, khanaen = self.ha_entropy(Xb, zb, ao_khanaen=1)
            self.entropy.append(entropy.kha)
            self.khanaen.append(khanaen)

    def ha_entropy(self, X, z, ao_khanaen=0):
        for c in self.chan[:-1]:
            X = c(X)
        if (ao_khanaen):
            return self.chan[-1](X, z), ((X.kha >= 0).flatten() == z).mean()
        return self.chan[-1](X, z)

    def param(self):
        p = []
        for c in self.chan:
            if (hasattr(c, 'param')):
                p.extend(c.param)
        return p

    def thamnai(self, X):
        for c in self.chan[:-1]:
            X = c(X)
        return (X.kha >= 0).flatten().astype(int)


squat = cd("dataSet/Squat")
curl = cd("dataSet/Barbell Curl")
pushup = cd('dataSet/Push Ups')
dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
deadlift = cd('dataSet/Deadlift')
cam = cd('dataSet/cam')
# target_names = np.array(['curl','pushup', 'squat', 'deadlift'], dtype='<U10')
target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')

path = [squat, curl,pushup, dumbbellShoulderPress, deadlift]
# path = [squat, pushup]
idc = 0
nxy, z = cd.allpath(path, idc)
x = cd.xx(nxy)
y = cd.yy(nxy)
z = cd.cen_z(z)
X = np.stack((x, y), axis=1)
z = np.array(z)
print(z)
print('Showdata OK...')

print('Start.....')
prasat = Prasat(m=[2,100,100,100,100,2],eta=0.01)
prasat.rianru(X,z,n_thamsam=1000)
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),200),np.linspace(X[:,1].min(),X[:,1].max(),200))
mX = np.array([mx.ravel(),my.ravel()]).T
mz = prasat.thamnai(mX).reshape(200,-1)
print(mz)
plt.gca(aspect=1,xlim=(X[:,0].min(),X[:,0].max()),ylim=(X[:,1].min(),X[:,1].max()))
plt.contourf(mx,my,mz,cmap='rainbow',alpha=0.2)
plt.scatter(X[:,0],X[:,1],50,c=z,edgecolor='k',cmap='rainbow')
plt.show()

plt.figure(figsize=[6,10])
ax1 = plt.subplot(211,xticks=[])
ax1.set_title(u'entropy')
ax2 = plt.subplot(212)
ax2.set_title(u'Score')
for Opt in [Sgd,Mmtsgd,Nag,Adagrad,Adadelta,Adam]:
    chan = [Affin(2,60,1),Sigmoid(),Affin(60,1,1),Sigmoid_entropy()]
    opt = Opt(chan[0].param+chan[2].param,eta=0.02)
    lis_entropy = []
    lis_khanaen = []
    for i in range(200):
        X_ = X
        for c in chan[:-1]:
            X_ = c(X_)
        lis_khanaen.append(((X_.kha.ravel()>0)==z).mean())
        entropy = chan[-1](X_,z)
        lis_entropy.append(entropy.kha)
        entropy.phraeyon()
        opt()
    si = np.random.random(3)
    ax1.plot(lis_entropy,color=si)
    ax2.plot(lis_khanaen,color=si)
plt.legend(['SGD','Momentum','NAG','AdaGrad','AdaDelta','Adam'],ncol=2)
plt.tight_layout()
plt.show()