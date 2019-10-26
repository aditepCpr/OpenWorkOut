import numpy as np
import pickle
import matplotlib.pyplot as plt
from ReadData import CreateData as cd


class D_tree:
    mz = []
    mx = []
    my = []
    mX = []

    def __init__(self, X, z):

        self.X = X
        self.z = z
        self.mz = self.mz
        self.mx = self.mx
        self.my = self.my
        self.mX = self.mX

    def d_tree(self):
        try:
            file_model = open('d_t.pkl', 'rb')
            stored_d_t = pickle.load(file_model)
            file_model.close()
        except IOError as e:
            print(e)
        nmesh = 2000
        self.mx, self.my = np.meshgrid(np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), nmesh),
                                       np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), nmesh))
        self.mX = np.stack([self.mx.ravel(), self.my.ravel()], 1)
        self.mz = stored_d_t.predict(self.mX).reshape(nmesh, nmesh)
        self.plottassimo(self.X, self.z, self.mx, self.my, self.mz)


    def plottassimo(self, X, z, mx, my, mz):
        plt.figure().gca(aspect=1, xlim=[mx.min(), mx.max()], ylim=[my.min(), my.max()])
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c=z, edgecolor='k', cmap='rainbow')
        plt.title("decision tree ")
        plt.contourf(mx, my, mz, alpha=0.4, cmap='rainbow', zorder=0)
        plt.show()

def tuni(mz, name):
    print(name)
    x0 = 0
    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    for h in mz:

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


squat = cd("dataSet/Squat")
curl = cd("dataSet/Barbell Curl")
pushup = cd('dataSet/Push Ups')
dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
deadlift = cd('dataSet/Deadlift')
cam = cd('dataSet/cam')
target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')


path = [squat]
idc = 0
nxy, z = cd.allpath(path, idc)
x = cd.xx(nxy)
y = cd.yy(nxy)
z = cd.cen_z(z)
X = np.stack((x, y), axis=1)
z = np.array(z)
xy_sta = (X-X.mean(0))/X.std(0)
print(xy_sta)
print('Showdata OK...')
plt.scatter(xy_sta[:, 0], xy_sta[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
plt.show()


print('Showdata OK...')


if __name__ == '__main__':
    print(__name__)

    d_tree = D_tree(X, z)
    d_tree.d_tree()
    tuni(d_tree.mz, 'decision tree')