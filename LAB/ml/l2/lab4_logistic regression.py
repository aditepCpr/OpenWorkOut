import numpy as np
import matplotlib.pyplot as plt
x = np.arange(1,31,2)
z = 2.5+x*0.5+np.random.randn(15)*0.5
# plt.gca(xlim=[0,31])
# plt.xlabel(u'เวลา (วัน)',fontname='Tahoma')
# plt.ylabel(u'ปริมาณอาหาร (กก.)',fontname='Tahoma')
# plt.scatter(x,z)
# plt.show()

# h = w*x+b
# sse = ((z-h)**2).sum()

from mpl_toolkits.mplot3d import Axes3D

plt.figure(figsize=[8,8])
ax = plt.axes([0,0,1,1],projection='3d',xlabel='b',ylabel='w',zlabel='SSE')
mb,mw = np.meshgrid(np.linspace(0,5,41),np.linspace(0,1,41))
sse = ((x*mw.ravel()[:,None]+mb.ravel()[:,None]-z)**2).sum(1).reshape(41,-1)
ax.plot_surface(mb,mw,sse,rstride=1,cstride=1,alpha=0.2)
plt.show()

import matplotlib as mpl
mb,mw = np.meshgrid(np.linspace(0,5,201),np.linspace(0,1,201))
sse = ((x*mw.ravel()[:,None]+mb.ravel()[:,None]-z)**2).sum(1).reshape(201,-1)
plt.gca(xlim=[0,5],ylim=[0,1])
plt.pcolormesh(mb,mw,sse,norm=mpl.colors.LogNorm(),cmap='gnuplot')
plt.colorbar(pad=0.01)
plt.show()


eta = 0.0002 # คืออัตราการเรียนรู้
n_thamsam = 10000 # จำนวนครั้งที่ทำซ้ำเพื่อเรียนรู้
w,b = 0,0 # น้ำหนักและไบแอสเริ่มต้น
wi = [w] # ลิตต์บันทุกค่าน้ำหนักและไบแอส
bi = [b]
h = w*x+b # คำนวณคำตอบโดนใช้ w และ  b ตอนแรก
for i in range(n_thamsam):
    w += 2*((z-h)*x).sum()*eta # ปรับค่าน้ำหนักและไบแอส
    b += 2*(z-h).sum()*eta
    wi += [w] # บันทึกค่าน้ำหนักและไบแอส
    bi += [b]
    h = w*x+b # คำนวณคำตอบโดยใช้ค่า w และ b ใหม่

plt.gca(xlim=[0,31])
plt.xlabel(u'เวลา (วัน)',fontname='Tahoma')
plt.ylabel(u'ปริมาณอาหาร (กก.)',fontname='Tahoma')
plt.scatter(x,z)
xsen = np.array([0,31])
ysen = xsen*w+b
plt.plot(xsen,ysen,'b')
plt.show()

bi = np.array(bi)
wi = np.array(wi)
plt.figure(figsize=[8,8])
ax = plt.axes([0,0,1,1],projection='3d',xlabel='b',ylabel='w',zlabel='SSE')
ssei = ((x*wi[:,None]+bi[:,None]-z)**2).sum(1)
ax.plot(bi,wi,ssei,'bo-')
mb,mw = np.meshgrid(np.linspace(0,3,201),np.linspace(0,1.2,201))
sse = ((x*mw.ravel()[:,None]+mb.ravel()[:,None]-z)**2).sum(1).reshape(201,-1)
ax.plot_surface(mb,mw,sse,rstride=5,cstride=5,alpha=0.2,color='b',edgecolor='k')

plt.figure(figsize=[10,5])
plt.pcolormesh(mb,mw,sse,norm=mpl.colors.LogNorm(),cmap='gnuplot')
plt.colorbar(pad=0.01)
plt.plot(bi,wi,'bo-')
plt.show()

eta = 0.0002
n_thamsam = 100000
d_yut = 1e-7 # ค่าความเปลี่ยนแปลงน้ำหนักและไบแอสสูงสุดที่จะให้หยุดได้
w,b = 0,0
h = w*x+b
for i in range(n_thamsam):
    dw = 2*((z-h)*x).sum()*eta
    db = 2*(z-h).sum()*eta
    w += dw
    b += db
    h = w*x+b
    if(np.abs(dw)and np.abs(db)<d_yut):
        break # หยุดเมื่อทั้ง dw และ db ต่ำกว่า d_yut

print('ทำซ้ำไป %d ครั้ง dw=%.3e, db=%.3e'%(i,dw,db))
# ทำซ้ำไป 7064 ครั้ง dw=-5.006e-09, db=9.993e-08

###  ลองใช้โจทย์ข้อเดิมแต่ปรับ eta ให้สูงขึ้น แล้วลองทำซ้ำดูแค่ ๑๐ ครั้ง จากนั้นดูค่า w, b และ SSE ที่เปลี่ยนแปลงไป
eta = 0.0003
n_thamsam = 10
w,b = 0,0
wi = [w]
bi = [b]
h = w*x+b
for i in range(n_thamsam):
    w += 2*((z-h)*x).sum()*eta
    b += 2*(z-h).sum()*eta
    wi += [w]
    bi += [b]
    h = w*x+b

bi = np.array(bi)
wi = np.array(wi)
plt.figure(figsize=[8,8])
ax = plt.axes([0,0,1,1],projection='3d',xlabel='b',ylabel='w',zlabel='SSE')
ssei = ((x*wi[:,None]+bi[:,None]-z)**2).sum(1)
ax.plot(bi,wi,ssei,'bo-')
mb,mw = np.meshgrid(np.linspace(bi.min(),bi.max(),201),np.linspace(wi.min(),wi.max(),201))
sse = ((x*mw.ravel()[:,None]+mb.ravel()[:,None]-z)**2).sum(1).reshape(201,-1)
ax.plot_surface(mb,mw,sse,rstride=5,cstride=5,alpha=0.2,color='b',edgecolor='k')
plt.show()