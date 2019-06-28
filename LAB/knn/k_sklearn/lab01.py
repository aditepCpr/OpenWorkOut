import codecs, json
import numpy as np
from sklearn import datasets
import os
import matplotlib.pyplot as plt


def train_classifier():
    path1 = [os.path.join("dataSet/curl", f) for f in os.listdir("dataSet/curl")]
    path2 = [os.path.join("dataSet/squat", f) for f in os.listdir("dataSet/squat")]

    # print(path)
    xy = []
    for kpose in path1:
        try:
            file_path = (kpose)
            file = codecs.open(file_path, 'r', encoding='utf-8').read()
            b_new = json.loads(file)
            x = np.array(b_new)
            xy.append(x)
        except IOError as e:
            print(e)

    xx = []
    for x in xy:
        npx = np.array(x)
        x = npx[:, 0:1]
        for x1 in x :
            for x2 in x1:
                xx.append(x2)
                # print(x2)
    yy = []
    for x in xy:
        npx = np.array(x)
        y = npx[:, 1:2]
        for y1 in y:
            for y2 in y1:
                yy.append(y2)
                # print(y2)

    xy = np.stack((xx,yy),axis=1)

    supxy = []
    for xxyy in xy :
            supxy.append(0)
    supp = np.asarray(supxy)
    #
    import matplotlib.pyplot as plt
    plt.figure(figsize=[6, 6])
    plt.gca(aspect=1)
    plt.scatter(xy[:,0], xy[:,1],c=supp, edgecolor='k')
    plt.show()

    xy = []
    for kpose in path2:
         try:
            file_path = (kpose)
            file = codecs.open(file_path, 'r', encoding='utf-8').read()
            b_new = json.loads(file)
            x = np.array(b_new)
            xy.append(x)
         except IOError as e:
            print(e)
    xx = []
    for x in xy:
        npx = np.array(x)
        x = npx[:, 0:1]
        for x1 in x :
            for x2 in x1:
                xx.append(x2)
                # print(x2)
    yy = []
    for x in xy:
        npx = np.array(x)
        y = npx[:, 1:2]
        for y1 in y:
            for y2 in y1:
                yy.append(y2)
                # print(y2)

    xy = np.stack((xx,yy),axis=1)

    supxy = []
    for xxyy in xy :
            supxy.append(1)
    supp = np.asarray(supxy)
    #
    import matplotlib.pyplot as plt
    plt.figure(figsize=[6, 6])
    plt.gca(aspect=1)
    plt.scatter(xy[:,0], xy[:,1],c=supp, edgecolor='k')
    plt.show()





    # plt.gca(aspect=1).scatter(xy[:, 0], xy[:, 1], c=supp, s=30, edgecolor='k', cmap='rainbow')
    # plt.show()
    # from sklearn.neighbors import KNeighborsClassifier as Knn
    # knn = Knn(n_neighbors=1,p=1)
    # print(xy[:100])
    # print(supp.size)
    # print(supp[:100])
    # knn.fit(xy,supp)
    #
    # nmesh = 200
    # mx,my = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), nmesh), np.linspace(xy[:, 1].min(), xy[:, 1].max(), nmesh))
    # mX = np.stack([mx.ravel(),my.ravel()],1)
    # mz = knn.predict(mX).reshape(nmesh,nmesh)
    # plt.gca(xlim=[xy[:, 0].min(), xy[:, 0].max()], ylim=[xy[:, 1].min(), xy[:, 1].max()], aspect=1)
    # plt.contourf(mx,my,mz,alpha=0.1,cmap='rainbow')
    # plt.contour(mx,my,mz,colors='#222222')
    # plt.scatter(xy[:, 0], xy[:, 1], c=supp, edgecolor='k', cmap='rainbow')
    # plt.show()

    # nmesh = 200
    # mx, my = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), nmesh),
    #                      np.linspace(xy[:, 1].min(), xy[:, 1].max(), nmesh))
    # mX = np.stack([mx.ravel(), my.ravel()], 1)
    # for i in [0, 1]:
    #     n = 3 + 27 * i
    #     knn = Knn(n)
    #     knn.fit(xy, supp)
    #     k = knn.kneighbors(xy)
    #     for j in [0, 1]:
    #         if (j == 1):
    #             mz = knn.predict_proba(mX)[:, 1].reshape(nmesh, nmesh)
    #         else:
    #             mz = knn.predict(mX).reshape(nmesh, nmesh)
    #         plt.subplot(221 + i + 2 * j, xlim=[xy[:, 0].min(), xy[:, 0].max()], ylim=[xy[:, 1].min(), xy[:, 1].max()],
    #                     aspect=1)
    #         plt.scatter(xy[:, 0], xy[:, 1], 10, c=supp, edgecolor='k', cmap='winter')
    #         plt.contourf(mx, my, mz, 100, cmap='winter', zorder=0)
    #         if (j == 0):
    #             plt.title('n=%d' % n)
    #         else:
    #             plt.ylabel('proba')
    # plt.show()

train_classifier()
# squat = train_classifier("dataSet/squat",1)

