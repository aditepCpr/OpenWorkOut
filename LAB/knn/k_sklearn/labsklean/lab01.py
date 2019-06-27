import numpy as np
from body.KeyPoints import *
from sklearn import datasets
def Lab(bodyKeypoints):
    X, z = datasets.make_blobs(n_samples=1000, n_features=4, centers=5)
    X = bodyKeypoints

    print(X.dtype)
    print(X.size)
    print(X.shape)
    print(X[:100])

    import matplotlib.pyplot as plt
    plt.figure(figsize=[6,6])
    plt.gca(aspect=1)
    plt.scatter(X[:,0],X[:,1],c=z,edgecolor='k')
    plt.show()

