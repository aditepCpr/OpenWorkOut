from createData import CreateData as cd
import numpy as np
import time
squat = cd("dataSet/squat")
curl = cd("dataSet/curl")
path = [squat,curl]
idc = 0


nxy,z = cd.allpath(path,idc)
xxx = cd.xx(nxy)
yyy = cd.yy(nxy)
z = cd.cen_z(z)
supperxy = np.stack((xxx,yyy),axis=1)


import matplotlib.pyplot as plt


plt.gca(aspect=1).scatter(supperxy[:, 0], supperxy[:, 1], c=z, s=30, edgecolor='k', cmap='rainbow')
plt.show()
from sklearn.neighbors import KNeighborsClassifier as Knn
knn = Knn(n_neighbors=1,p=1)
knn.fit(supperxy, z)

nmesh = 200
mx,my = np.meshgrid(np.linspace(supperxy[:, 0].min(), supperxy[:, 0].max(), nmesh), np.linspace(supperxy[:, 1].min(), supperxy[:, 1].max(), nmesh))
mX = np.stack([mx.ravel(),my.ravel()],1)
mz = knn.predict(mX).reshape(nmesh,nmesh)
plt.gca(xlim=[supperxy[:, 0].min(), supperxy[:, 0].max()], ylim=[supperxy[:, 1].min(), supperxy[:, 1].max()], aspect=1)
plt.contourf(mx,my,mz,alpha=0.1,cmap='rainbow')
plt.contour(mx,my,mz,colors='#222222')
plt.scatter(supperxy[:, 0], supperxy[:, 1], c=z, edgecolor='k', cmap='rainbow')
plt.show()

for al in ['ball_tree','kd_tree','brute','auto']:
    t1 = time.time()
    knn = Knn(algorithm=al)
    knn.fit(supperxy,z)
    knn.predict(supperxy)
    print(u'%s: %.3f วินาที'%(al,time.time()-t1))


for j in [1,2,3]:
    t1 = time.time()
    Knn(n_jobs=j).fit(supperxy,z).predict(supperxy)
    print(u'n_jobs=%s: %.3f วินาที'%(j,time.time()-t1))

knn = Knn(n_neighbors=3)
knn.fit(supperxy,z)
plt.gca(aspect=1)
plt.scatter(supperxy[:,0],supperxy[:,1],c=z,edgecolor='k',cmap='summer')
plt.show()
k = knn.kneighbors(supperxy)
print(k[0])
print(k[1])
for i in range(8):
    print(', '.join(['%d > %.2f'%(k[1][i][j],k[0][i][j]) for j in range(3)]))

print(type(knn.kneighbors_graph(supperxy))) # ให้แสดงชนิด
print(knn.kneighbors_graph(supperxy).toarray()) # แปลงเป็นอาเรย์ธรรมดา

X = supperxy

nmesh = 200
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),nmesh),np.linspace(X[:,1].min(),X[:,1].max(),nmesh))
mX = np.stack([mx.ravel(),my.ravel()],1)
for i in [0,1]:
    n = 3+27*i
    knn = Knn(n)
    knn.fit(X,z)
    k = knn.kneighbors(X)
    for j in [0,1]:
        if(j==1):
            mz = knn.predict_proba(mX)[:,1].reshape(nmesh,nmesh)
        else:
            mz = knn.predict(mX).reshape(nmesh,nmesh)
        plt.subplot(221+i+2*j,xlim=[X[:,0].min(),X[:,0].max()],ylim=[X[:,1].min(),X[:,1].max()],aspect=1)
        plt.scatter(X[:,0],X[:,1],10,c=z,edgecolor='k',cmap='winter')
        plt.contourf(mx,my,mz,100,cmap='winter',zorder=0)
        if(j==0):
            plt.title('n=%d'%n)
        else:
            plt.ylabel('proba')
plt.show()