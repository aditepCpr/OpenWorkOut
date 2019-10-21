from createData import CreateData as cd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as Knn

squat = cd("dataSet/Squat")
curl = cd("dataSet/Barbell Curl")
pushup = cd('dataSet/Push Ups')
dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
deadlift = cd('dataSet/Deadlift')
cam = cd('dataSet/cam')
# target_names = np.array(['curl','pushup', 'squat', 'deadlift'], dtype='<U10')
target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')

path = [squat, curl,pushup, dumbbellShoulderPress, deadlift]
# path = [squat, curl]
idc = 0
nxy, z = cd.allpath(path, idc)
xxx = cd.xx(nxy)
yyy = cd.yy(nxy)
z = cd.cen_z(z)
supperxy = np.stack((xxx, yyy), axis=1)
print('Showdata OK...')

path2 = [curl]
idcz = 0
nxyz, zz = cd.allpath(path2, idcz)
xxxz = cd.xx(nxyz)
yyyz = cd.yy(nxyz)
zz2 = cd.cen_z(zz)
supperxyz = np.stack((xxxz, yyyz), axis=1)



def show2():
    knn = Knn(n_neighbors=1, p=1)
    knn.fit(supperxy, z)
    nmesh = 200
    mx, my = np.meshgrid(np.linspace(supperxyz[:, 0].min(), supperxyz[:, 0].max(), nmesh),
                         np.linspace(supperxyz[:, 1].min(), supperxyz[:, 1].max(), nmesh))
    mX = np.stack([mx.ravel(), my.ravel()], 1)
    mz = knn.predict(mX).reshape(nmesh, nmesh)
    print(mX)
    # print(target_names[mz])

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
    print(target_names[0], 'x0', x0)
    print(target_names[1], 'x1', x1)
    print(target_names[2], 'x2', x2)
    print(target_names[3], 'x3', x3)
    print(target_names[4], 'x4', x4)
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

    plt.gca(xlim=[supperxyz[:, 0].min(), supperxyz[:, 0].max()], ylim=[supperxyz[:, 1].min(), supperxyz[:, 1].max()],
            aspect=1)
    plt.contourf(mx, my, mz, alpha=0.1, cmap='rainbow')
    plt.contour(mx, my, mz, colors='#222222')
    plt.scatter(supperxyz[:, 0], supperxyz[:, 1], c=zz2, edgecolor='k', cmap='rainbow')
    plt.show()


def show4():
    from sklearn import decomposition
    nmesh = 200
    pca = decomposition.PCA(n_components=2)
    pca.fit(supperxy)
    X = pca.transform(supperxy)

    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=z)
    plt.show()

    from sklearn.cluster import KMeans
    indexpath = int(len(path))
    km = KMeans(indexpath)
    mX = km.fit(X)
    # print(km.cluster_centers_)
    mz = km.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1], c=z)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='o', s=100)
    plt.show()

    pca2 = decomposition.PCA(n_components=2)
    pca2.fit(supperxyz)
    X2 = pca.transform(supperxyz)
    plt.scatter(X2[:, 0], X2[:, 1], c=zz2)
    plt.show()


    indexpath2 = int(len(path2))
    km2 = KMeans(indexpath2)
    mX2 = km2.fit(X2)
    mz2 = km2.cluster_centers_
    plt.scatter(X2[:, 0], X2[:, 1], c=zz2)
    plt.scatter(km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1], marker='o', s=100)
    plt.show()

    ZZZ = np.arange(int(len(path)))
    XZ = mz[:, 0]
    YZ = mz[:, 1]
    MZXY = np.stack((XZ, YZ), axis=1)

    XZ2 = mz2[:, 0]
    YZ2 = mz2[:, 1]
    MZXY2 = np.stack((XZ2, YZ2), axis=1)

    knn = Knn(n_neighbors=1, p=1)
    knn.fit(MZXY, ZZZ)

    mx, my = np.meshgrid(np.linspace(MZXY[:, 0].min(), MZXY[:, 0].max(), nmesh),
                         np.linspace(MZXY[:, 1].min(), MZXY[:, 1].max(), nmesh))
    mx2, my2 = MZXY2[:, 0].min(), MZXY2[:, 1].max()
    mX1 = np.stack([mx.ravel(), my.ravel()], 1)
    MX2 = np.stack([mx2.ravel(), my2.ravel()], 1)
    mz1 = knn.predict(mX1).reshape(nmesh, nmesh)
    mz2 = knn.predict(MX2)
    ZMZ = [0]
    print(ZMZ)

    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    for h in mz2:
        print(h)
        if h == 0:
            x0 += 1
        elif h == 1:
            x1 += 1
        elif h == 2:
            x2 += 1
        elif h == 3:
            x3 += 1
        elif h == 4:
            x4 += 1

        max_h = max(x0, x1, x2, x3, x4)
        print(target_names[0], 'x0', x0)
        print(target_names[1], 'x1', x1)
        print(target_names[2], 'x2', x2)
        print(target_names[3], 'x3', x3)
        print(target_names[4], 'x4', x4)
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


    plt.gca(xlim=[mX1[:, 0].min(), mX1[:, 0].max()], ylim=[mX1[:, 1].min(), mX1[:, 1].max()],
            aspect=1)
    plt.contourf(mx, my, mz1, alpha=0.1, cmap='rainbow')
    plt.contour(mx, my, mz1, colors='#222222')
    plt.scatter(MX2[:, 0], MX2[:, 1], c=ZMZ, edgecolor='k', cmap='rainbow')
    plt.show()


def show5():
    print(show5.__name__)
    knn = Knn(n_neighbors=1, p=1)
    knn.fit(supperxy, z)
    nmesh = 200
    mx, my = np.meshgrid(np.linspace(supperxy[:, 0].min(), supperxy[:, 0].max(), nmesh),
                        np.linspace(supperxy[:, 1].min(), supperxy[:, 1].max(), nmesh))
    mX = np.stack([mx.ravel(), my.ravel()], 1)
    mz = knn.predict(mX).reshape(nmesh, nmesh)

    plt.gca(xlim=[supperxy[:, 0].min(), supperxy[:, 0].max()], ylim=[supperxy[:, 1].min(), supperxy[:, 1].max()],
           aspect=1)
    plt.contourf(mx, my, mz, alpha=0.1, cmap='rainbow')
    plt.contour(mx, my, mz, colors='#222222')
    plt.scatter(supperxy[:, 0], supperxy[:, 1], c=z, edgecolor='k', cmap='rainbow')
    plt.show()

    print(mX)

    from sklearn import decomposition
    pca = decomposition.PCA(n_components=2)
    pca.fit(mX)
    X = pca.transform(mX)
    print(X.size,X.shape)
    print(mz.size,mz.shape)
    plt.scatter(mx, my, c=mz)
    plt.show()

    from sklearn.cluster import KMeans
    indexpath = int(len(path))
    km = KMeans(indexpath)
    mX = km.fit(X)
    # print(km.cluster_centers_)
    mz = km.cluster_centers_
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='o', s=100)
    plt.show()

    pca2 = decomposition.PCA(n_components=2)
    pca2.fit(supperxyz)
    X2 = pca.transform(supperxyz)
    plt.scatter(X2[:, 0], X2[:, 1], c=zz2)
    plt.show()

    indexpath2 = int(len(path2))
    km2 = KMeans(indexpath2)
    mX2 = km2.fit(X2)
    mz2 = km2.cluster_centers_
    plt.scatter(X2[:, 0], X2[:, 1], c=zz2)
    plt.scatter(km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1], marker='o', s=100)
    plt.show()

    ZZZ2 = np.arange(int(len(path)))
    XZ2 = mz[:, 0]
    YZ2 = mz[:, 1]
    MZXY2 = np.stack((XZ2, YZ2), axis=1)

    ZZZ3 = [0]
    XZ3 = mz2[:, 0]
    YZ3 = mz2[:, 1]
    MZXY3 = np.stack((XZ3, YZ3), axis=1)

    print(MZXY2)

    knn2 = Knn(n_neighbors=1, p=1)
    knn2.fit(MZXY2, ZZZ2)
    mx2, my2 = np.meshgrid(np.linspace(MZXY2[:, 0].min(), MZXY2[:, 0].max(), nmesh),
                             np.linspace(MZXY2[:, 1].min(), MZXY2[:, 1].max(), nmesh))
    MX2 = np.stack([mx2.ravel(), my2.ravel()], 1)
    mz2 = knn2.predict(MX2).reshape(nmesh, nmesh)

    mx3, my3 = MZXY3[:, 0].min(), MZXY3[:, 1].max()
    MX3 = np.stack([mx3.ravel(), my3.ravel()], 1)
    mz3 = knn.predict(MX3)

    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    for h in mz3:
        print(h)
        if h == 0:
            x0 += 1
        elif h == 1:
            x1 += 1
        elif h == 2:
            x2 += 1
        elif h == 3:
            x3 += 1
        elif h == 4:
            x4 += 1

        max_h = max(x0, x1, x2, x3, x4)
        print(target_names[0], 'x0', x0)
        print(target_names[1], 'x1', x1)
        print(target_names[2], 'x2', x2)
        print(target_names[3], 'x3', x3)
        print(target_names[4], 'x4', x4)
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


    plt.gca(xlim=[MX2[:, 0].min(), MX2[:, 0].max()], ylim=[MX2[:, 1].min(), MX2[:, 1].max()],
           aspect=1)
    plt.contourf(mx2, my2, mz2, alpha=0.1, cmap='rainbow')
    plt.contour(mx2, my2, mz2, colors='#222222')
    plt.scatter(MX3[:, 0], MX3[:, 1], c=ZZZ3, edgecolor='k', cmap='rainbow')
    plt.show()

if __name__ == '__main__':
    # show2()
    show4()
    # show5()