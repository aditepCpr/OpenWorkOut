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
    import json as js
    try:
        file_path = ('dataSet/keypose.10.json')
        file = open(file_path, 'r')
        Bodykey = file.read()
        print(Bodykey)
    except IOError as e :
        print(e)

    for i in Bodykey :
        print(i)

    #
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=[6,6])
    # plt.gca(aspect=1)
    # plt.scatter(X[:,0],X[:,1],c=z,edgecolor='k')
    # plt.show()


try:
    file_path = ('~dataSet/keypose.10.json')
    file = open(file_path, 'r')
    Bodykey = file.read()
    print(Bodykey)
except IOError as e:
    print(e)

