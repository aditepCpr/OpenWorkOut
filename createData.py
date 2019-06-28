import codecs, json
import numpy as np
import os

xy = None
xx = None
yy = None
class CreateData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.idc = 1
        self.xy = xy
        self.xx = xx

    def train_classifier(self):
        path = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]

        # print(path)
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
        for p in path:
            oxy = p.train_classifier()
            xx = CreateData.xx(oxy)
            yy = CreateData.yy(oxy)
            oxy = CreateData.xy(xx, yy)
            z = CreateData.c(xx, yy, idc)
            Z = np.array(z[:])
            idc = + 1
            nxy.append(oxy)
            Zc.append(Z)
        return nxy,Zc


