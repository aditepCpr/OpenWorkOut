import numpy as np
import matplotlib.pyplot as plt

def h(X):
    a = np.dot(X, w) + b
    return (a >= 0).astype(int)


w = np.array([0, 0.])
b = 0.1

X = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1]
])
z = np.array([0,0,0,1])

# วาดภาพแสดง
plt.gca(aspect=1)
plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='r',marker='D',cmap='hot')
plt.show()

eta = 0.2 # อัตราการเรียนรู้
print('เริ่มต้น: h(x)=%s, w=%s, b=%s'%(h(X),w,b))
for j in range(100): # ให้ทำซ้ำสูงสุด 100 ครั้ง
    for i in range(4):
        z_h = z[i] - h(X[i])
        dw = eta*z_h*X[i]
        db = eta*z_h
        w += dw
        b += db
        print('รอบ %d.%d: h(x)=%s, w=%s, b=%s, Δw=%s, Δb=%s'%(j+1,i+1,h(X),w,b,dw,db))
        # mx, my = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        # mX = np.array([mx.ravel(), my.ravel()]).T
        # mz = h(mX).reshape(200, -1)
        # plt.gca(aspect=1, xticks=[0, 1], yticks=[0, 1])
        # plt.contourf(mx, my, mz, cmap='summer', vmin=0, vmax=1)
        # plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='r', marker='D', cmap='hot')
        # plt.show()
    if(np.all(h(X)==z)): # ถ้าผลลัพธ์ถูกต้องทั้งหมดก็ให้เสร็จสิ้นการเรียนรู้
        break


mx,my = np.meshgrid(np.linspace(-0.5,1.5,200),np.linspace(-0.5,1.5,200))
mX = np.array([mx.ravel(),my.ravel()]).T
mz = h(mX).reshape(200,-1)
plt.gca(aspect=1,xticks=[0,1],yticks=[0,1])
plt.contourf(mx,my,mz,cmap='summer',vmin=0,vmax=1)
plt.scatter(X[:,0],X[:,1],100,c=z,edgecolor='r',marker='D',cmap='hot')
plt.show()
