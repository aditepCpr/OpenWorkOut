import numpy as np

def h(X):
    w = np.array([20,50])
    b = -1000
    a = (w*X).sum() + b
    return  int(a>=0)

X = np.array([20,10])
print(h(X))
X = np.array([14,15])
print((h(X)))