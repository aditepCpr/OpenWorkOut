import numpy as np

def h(X):
    w = np.array([1,1])
    b = -0.9
    a = np.dot(X,w) + b
    return (a>=0).astype(int)

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

print(h(X))

import matplotlib.pyplot as plt
mx,my = np.meshgrid(np.linspace(-0.5,1.5,200),np.linspace(-0.5,1.5,200))
mX = np.array([mx.ravel(),my.ravel()]).T
mz = h(mX).reshape(200,-1)
plt.gca(aspect=1)
plt.contourf(mx,my,mz,cmap='summer')
plt.scatter(X[:,0],X[:,1],100,c=h(X),edgecolor='r',marker='D',cmap='hot')
plt.show()