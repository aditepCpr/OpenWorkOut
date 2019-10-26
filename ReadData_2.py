import codecs, json
import numpy as np
import os

xy = None
xx = None
yy = None
class CreateData2:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.idc = 1
        self.xy = xy
        self.xx = xx
    # อ่านข้อมูลจากไฟล์
    def create_sum_xyz(self):
        path = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]
        xy = []
        for kpose in path:
            try:
                file_path = (kpose)
                file = codecs.open(file_path, 'r', encoding='utf-8').read()
                b_new = json.loads(file)
                x = np.array(b_new)
                xy.append(x)
            except IOError as e:
                print(e)
        return xy

    def xx_x(xy):
        xx = []
        xxx =[]
        i = 0
        for x in xy:
            npx = np.array(x)
            x = npx[:, 0:1]
            for x1 in x:
                for x2 in x1:
                    xx.append(x2)
                    i += 1
                    if i == 14:
                        xxx.append(sum(xx)/14)
                        xx.clear()
                        i = 0
        return xxx

    def yy_y(xy):
        yy = []
        yyy = []
        i = 0
        for x in xy:
            npx = np.array(x)
            y = npx[:, 1:2]
            for y1 in y:
                for y2 in y1:
                    yy.append(y2)
                    i += 1
                    if i == 14:
                        yyy.append(sum(yy) / 14)
                        yy.clear()
                        i = 0
        return yyy

    def xx(xy):
        xx = []
        for x in xy:
            npx = np.array(x)
            x = npx[:, 0:1]
            for x1 in x:
                for x2 in x1:
                    xx.append(x2)
                    # print(x2)
        return xx


    def yy(xy):
        yy = []
        for x in xy:
            npx = np.array(x)
            y = npx[:, 1:2]
            for y1 in y:
                for y2 in y1:
                    yy.append(y2)
                    # print(y2)
        return yy

    def c(xx,yy,idc):
        xy = np.stack((xx, yy), axis=1)
        supxy = []
        for xxyy in xy:
            supxy.append(idc)
        supp = np.asarray(supxy)
        return supp

    def xy(xx,yy):
        xy = np.stack((xx, yy), axis=1)
        return xy



    def cen_z(z):
        yz = []
        for y in z:
            yz2 = np.array(y)
            y = yz2[:]
            for y1 in y:
                yz.append(y1)
        return yz

    def allpath(path,idc):
        nxy = []
        Zc = []
        XY = []
        for p in path:
            oxy = p.create_sum_xyz()
            xx = CreateData2.xx(oxy)
            yy = CreateData2.yy(oxy)
            oxy = CreateData2.xy(xx, yy)
            nxy.append(oxy)
            idc += 1
            xxx = CreateData2.xx_x(nxy)
            yyy = CreateData2.yy_y(nxy)
            xy = CreateData2.xy(xxx, yyy)
            XY.append(xy)
            z = CreateData2.c(xxx, yyy, idc)
            Z = np.array(z[:])
            Zc.append(Z)
        return XY,Zc

# from ReadData_2 import CreateData2 as cd
# import numpy as np
# import time
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier as Knn
# from unagi import Affin, Softmax_entropy, Sigmoid_entropy, ha_1h
# from unagi import Sigmoid, Relu, Lrelu, Prelu, Elu, Selu, Tanh, Softsign, Softplus
# from sklearn.ensemble import RandomForestClassifier as Rafo

# if __name__ == "__main__":
#
#     squat = cd("dataSet/Squat")
#     curl = cd("dataSet/Barbell Curl")
#     pushup = cd('dataSet/Push Ups')
#     dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
#     deadlift = cd('dataSet/Deadlift')
#     cam = cd('dataSet/cam')
#     # target_names = np.array(['curl','pushup', 'squat', 'deadlift'], dtype='<U10')
#     target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')
#     # target_names = np.array(['squat', 'pushup', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')
#
#     path = [squat, curl, pushup, dumbbellShoulderPress, deadlift]
#     idc = 0
#     XY, z = cd.allpath(path, idc)
#     xx = cd.xx(XY)
#     yy = cd.yy(XY)
#     z = cd.cen_z(z)
#     X = np.stack((xx, yy), axis=1)
#     z = np.array(z)
#
#     print(len(z))
#     print(len(X))
#     # print('Showdata OK...')
#     plt.scatter(X[:, 0], X[:, 1], 30, c=z, edgecolor='k', cmap='rainbow')
#     plt.show()

    # deadlift = cd('dataSet/Deadlift')
    # read_data = cd.create_allxyz(deadlift)
    # x = cd.xx(read_data)
    # y = cd.yy(read_data)
    # print(x)
    # print(y)
