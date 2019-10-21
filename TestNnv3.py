import numpy as np

class Phueanban:
    def __init__(self,nk=5):
        self.nk = nk # จำนวนเพื่อนบ้านที่จะพิจารณา

    def rianru(self,X,z):
        self.X = X # เก็บข้อมูลตำแหน่ง
        self.z = z # เก็บข้อมูลการแบ่งกลุ่ม
        self.n_klum = z.max()+1 # จำนวนกลุ่ม

    def thamnai(self,X):
        n = len(X) # จำนวนข้อมูลที่จะคำนวณหา
        n_batch = int(np.ceil(1000000./X.shape[1]/len(self.X)))
        z = np.empty(n,dtype=int)
        for c in range(0,n,n_batch):
            Xn = X[c:c+n_batch]
            n_Xn = len(Xn)
            raya2 = ((Xn[None]-self.X[:,None])**2).sum(2)
            klum_thi_klai = self.z[raya2.argsort(0)]
            n_nai_klum = np.stack([(klum_thi_klai[:self.nk]==k).sum(0) for k in range(self.n_klum)])
            mi_maksut = n_nai_klum.max(0)
            maksutmai = (n_nai_klum==mi_maksut)

            for i in range(n_Xn):
                for j in range(self.nk):
                    k = klum_thi_klai[j,i]
                    if(maksutmai[k,i]):
                        z[i+c] = k
                        break
        return z

from sklearn import datasets
from sklearn.model_selection import train_test_split
from createData import CreateData as cd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as Knn
from unagi import Affin, Sigmoid, Relu, Softmax_entropy, Sigmoid_entropy, ha_1h
squat = cd("dataSet/Squat")
curl = cd("dataSet/Barbell Curl")
pushup = cd('dataSet/Push Ups')
dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
deadlift = cd('dataSet/Deadlift')
cam = cd('dataSet/cam')
# target_names = np.array(['curl','pushup', 'squat', 'deadlift'], dtype='<U10')
target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')
# target_names = np.array(['squat', 'pushup', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')

path = [squat, curl,pushup, dumbbellShoulderPress, deadlift]
# path = [squat, curl]
idc = 0
nxy, z = cd.allpath(path, idc)
x = cd.xx(nxy)
y = cd.yy(nxy)
z = cd.cen_z(z)
X = np.stack((x, y), axis=1)
z = np.array(z)
np.random.seed(0)
X_fuek,X_truat,z_fuek,z_truat = train_test_split(X,z,test_size=200/70000)

pb = Phueanban(nk=1)
pb.rianru(X_fuek,z_fuek)
zz = pb.thamnai(X_fuek[:200]) # ทำนายข้อมูลฝึก ดึงมาแค่ 200 ตัว ให้เท่ากับข้อมูลตรวจสอบ
print((zz==z_fuek[:200]).mean()) # ได้ 1.0
zz = pb.thamnai(X_truat) # ทำนายข้อมูลตรวจสอบ
print((zz==z_truat).mean()) # ได้ 0.975