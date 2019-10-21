from createData import CreateData as cd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as Knn
from unagi import Affin, Sigmoid, Relu, Softmax_entropy, Sigmoid_entropy, ha_1h
from sklearn.linear_model import LogisticRegression as Lori

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
print('Showdata OK...')
plt.scatter(X[:, 0], X[:, 1], 50, c=z, edgecolor='k', cmap='rainbow')
plt.show()

path2 = [cam]
idc2 = 0
nxy2, z2 = cd.allpath(path2, idc2)
x2 = cd.xx(nxy2)
y2 = cd.yy(nxy2)
z2 = cd.cen_z(z2)
X2 = np.stack((x2, y2), axis=1)
plt.scatter(X2[:, 0], X2[:, 1], 50, c=z2, edgecolor='k', cmap='rainbow')
plt.show()


lori = Lori()
lori.fit(X,z)

z2 = lori.predict(X2)
plt.gca(aspect=1)
plt.scatter(X[:,0],X[:,1],c=z,s=30,alpha=0.3,edgecolor='k',cmap='brg')
plt.scatter(X2[:,0],X2[:,1],c=z2,s=700,marker='*',edgecolor='k',cmap='brg')
plt.show()

nmesh = 1000 # สร้างจุดที่จะให้ทำนาย เป็นตาราง 100x100 รอบบริเวณนี้
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),nmesh),
                    np.linspace(X[:,1].min(),X[:,1].max(),nmesh))

# ปรับให้เรียงต่อกันเป็นอาเรย์หนึ่งมิติ แล้วรวมค่า x และ y เข้าเป็นอาเรย์เดียว
mX = np.stack([mx.ravel(),my.ravel()],1)
mz = lori.predict(mX) # ทำการทำนาย
mz = mz.reshape(nmesh,nmesh) # เปลี่ยนรูปกลับเป็นอาเรย์สองมิติ
plt.figure()
plt.gca(aspect=1)
plt.pcolormesh(mx,my,mz,alpha=0.6,cmap='brg') # วาดสีพื้น
plt.scatter(X[:,0],X[:,1],c=z,s=30,edgecolor='k',cmap='brg') # วาดจุดข้อมูล
plt.show()