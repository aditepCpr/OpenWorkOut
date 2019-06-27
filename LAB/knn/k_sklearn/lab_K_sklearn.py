import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

X,z = datasets.make_blobs(n_samples=100,centers=4,cluster_std=0.8,random_state=0)
plt.gca(aspect=1).scatter(X[:,0],X[:,1],c=z,s=30,edgecolor='k',cmap='rainbow')
plt.show()

#
# # การแบ่งเขตด้วยวิธีการเพื่อนบ้านใกล้สุด k
#
# from sklearn.neighbors import KNeighborsClassifier as Knn
# knn = Knn(n_neighbors=1,p=1)
# knn.fit(X,z)
#
# nmesh = 200
# mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),nmesh),np.linspace(X[:,1].min(),X[:,1].max(),nmesh))
# mX = np.stack([mx.ravel(),my.ravel()],1)
# mz = knn.predict(mX).reshape(nmesh,nmesh)
# plt.gca(xlim=[X[:,0].min(),X[:,0].max()],ylim=[X[:,1].min(),X[:,1].max()],aspect=1)
# plt.contourf(mx,my,mz,alpha=0.1,cmap='rainbow')
# plt.contour(mx,my,mz,colors='#222222')
# plt.scatter(X[:,0],X[:,1],c=z,edgecolor='k',cmap='rainbow')
# plt.show()
#
# # เปรียบเทียบอัลกอริธึมดูบ้าง ลองเขียนแบบนี้ ข้อมูลมี 1000 ตัว แล้วตัวแปรต้นมี 10 ชนิด
# import time
# X,z = datasets.make_blobs(n_samples=10000,n_features=10,centers=4,random_state=0)
# for al in ['ball_tree','kd_tree','brute','auto']:
#     t1 = time.time()
#     knn = Knn(algorithm=al)
#     knn.fit(X,z)
#     knn.predict(X)
#     print(u'%s: %.3f วินาที'%(al,time.time()-t1))
#
#
# # ลองทดสอบเรื่องจำนวนจ็อบ
# X,z = datasets.make_blobs(n_samples=20000,n_features=20,random_state=0)
# for j in [1,2,3]:
#     t1 = time.time()
#     Knn(n_jobs=j).fit(X,z).predict(X)
#     print(u'n_jobs=%s: %.3f วินาที'%(j,time.time()-t1))
#
# ###############################################################################################
#
# X,z = datasets.make_blobs(n_samples=8,centers=2,random_state=5)
# knn = Knn(n_neighbors=3)
# knn.fit(X,z)
# plt.gca(aspect=1)
# plt.scatter(X[:,0],X[:,1],c=z,edgecolor='k',cmap='summer')
# plt.show()
# k = knn.kneighbors(X)
# print(k[0])
# print(k[1])
# for i in range(8):
#     print(', '.join(['%d > %.2f'%(k[1][i][j],k[0][i][j]) for j in range(3)]))
#
#
# print(type(knn.kneighbors_graph(X))) # ให้แสดงชนิด
# print(knn.kneighbors_graph(X).toarray()) # แปลงเป็นอาเรย์ธรรมดา
#
#
# X,z = datasets.make_blobs(n_samples=200,centers=2,cluster_std=2.5,random_state=12)
# nmesh = 200
# mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),nmesh),np.linspace(X[:,1].min(),X[:,1].max(),nmesh))
# mX = np.stack([mx.ravel(),my.ravel()],1)
# for i in [0,1]:
#     n = 3+27*i
#     knn = Knn(n)
#     knn.fit(X,z)
#     k = knn.kneighbors(X)
#     for j in [0,1]:
#         if(j==1):
#             mz = knn.predict_proba(mX)[:,1].reshape(nmesh,nmesh)
#         else:
#             mz = knn.predict(mX).reshape(nmesh,nmesh)
#         plt.subplot(221+i+2*j,xlim=[X[:,0].min(),X[:,0].max()],ylim=[X[:,1].min(),X[:,1].max()],aspect=1)
#         plt.scatter(X[:,0],X[:,1],10,c=z,edgecolor='k',cmap='winter')
#         plt.contourf(mx,my,mz,100,cmap='winter',zorder=0)
#         if(j==0):
#             plt.title('n=%d'%n)
#         else:
#             plt.ylabel('proba')
# plt.show()