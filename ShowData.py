from createData import CreateData as cd
import numpy as np
squat = cd("dataSet/squat")
curl = cd("dataSet/curl")
path = [squat,curl]
idc = 0


nxy,z = cd.allpath(path,idc)
print(z)
xxx = cd.xx(nxy)
yyy = cd.yy(nxy)
z = cd.cen_z(z)
supperxy = np.stack((xxx,yyy),axis=1)



import matplotlib.pyplot as plt

plt.gca(aspect=1).scatter(supperxy[:,0],supperxy[:,1],c=z,s=30,edgecolor='k',cmap='rainbow')
plt.show()



import matplotlib.pyplot as plt


plt.gca(aspect=1).scatter(supperxy[:, 0], supperxy[:, 1], c=z, s=30, edgecolor='k', cmap='rainbow')
plt.show()
from sklearn.neighbors import KNeighborsClassifier as Knn
knn = Knn(n_neighbors=1,p=1)
print(nxy[:100])
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