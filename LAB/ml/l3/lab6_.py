import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)
X = np.random.normal(0,0.5,[40,2])
X[:20] += 1.5
z = np.zeros(40)
z[20:] += 1

plt.gca(aspect=1)
plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='k',alpha=0.6,cmap='coolwarm')
plt.show()
def sigmoid(x):
    return 1/(1+np.exp(-x))

def ha_entropy(z,h):
    return -(z*np.log(h)+(1-z)*np.log(1-h)).mean()
w = np.array([0,0.])

b = 0
eta = 0.1
entropy = []
khanaen = []
for o in range(1000):
    a = np.dot(X,w) + b
    h = sigmoid(a)
    ga = (h-z)/len(z)
    gw = np.dot(X.T,ga)
    gb = ga.sum()
    w -= eta*gw
    b -= eta*gb
    entropy.append(ha_entropy(z,h)) # เอนโทรปี
    khanaen.append(((a>=0)==z).mean()) # คะแนน (สัดส่วนที่ทายถูก)

lins = np.linspace(-0.5,1.5,200)
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),200),np.linspace(X[:,1].min(),X[:,1].max(),200))
mX = np.array([mx.ravel(),my.ravel()]).T
mh = np.dot(mX,w) + b
mz = (mh>=0).astype(int).reshape(200,-1)
plt.gca(aspect=1,xlim=(X[:,0].min(),X[:,0].max()),ylim=(X[:,1].min(),X[:,1].max()))
plt.contourf(mx,my,mz,cmap='coolwarm',alpha=0.2)
plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='k',alpha=0.6,cmap='coolwarm')
plt.show()

plt.subplot(211,xticks=[])
plt.plot(entropy,'C4')
plt.ylabel(u'เอนโทรปี',family='Tahoma',size=14)
plt.subplot(212)
plt.plot(khanaen,'C4')
plt.ylabel(u'คะแนน',family='Tahoma',size=14)
plt.xlabel(u'จำนวนรอบ',family='Tahoma',size=14)
plt.show()