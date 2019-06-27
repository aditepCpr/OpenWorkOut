import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from LAB.knn.knnBasic.Phueanban import Phueanban
np.random.seed(10)
n_klum = 3
tamnaeng,klum = datasets.make_blobs(n_samples=120,n_features=2,centers=9,cluster_std=0.8)
klum %= n_klum
plt.figure(figsize=[6,6])
plt.gca(aspect=1)
plt.scatter(tamnaeng[:,0],tamnaeng[:,1],c=klum,edgecolor='k',cmap='viridis')
plt.show()

X = np.array([-1.5,6]) # สร้างจุดขึ้นมาจุดหนึ่ง
raya2 = ((tamnaeng-X)**2).sum(1) # คำนวณระยะสู่แต่ละจุดกำลังสอง (ไม่จำเป็นต้องถอดรากเพราะแค่จะใช้เปรียบเทียบ)
i = raya2.argmin() # ดัชนีของจุดที่ระยะน้อยที่สุด
print(klum[i]) # คำตอบ 0


# การหาจุดที่ใกล้สุดเป็นอันดับต้นๆอาจใช้คำสั่ง argsort
nk = 5 # จำนวนจุดที่จะพิจารณา
raya2 = ((tamnaeng-X)**2).sum(1)
an_thi_klai = raya2.argsort() # หาดัชนีของจุดที่มีค่าใกล้สุดไล่มาเรื่อยๆ
klum_thi_klai = klum[an_thi_klai] # ดูว่าดัชนีที่ว่านี้คือกลุ่มไหนบ้าง
n_nai_klum = np.array([(klum_thi_klai[:nk]==k).sum() for k in range(n_klum)]) # ดูว่า ๕ อันแรกที่ใกล้สุดมีกลุ่มไหนอยู่กี่ตัว
mi_maksut = n_nai_klum.max() # จำนวนมากที่สุดมีเท่าไหร่
maksutmai = n_nai_klum==mi_maksut # ดูว่ากลุ่มนี้มากสุดหรือไม่
for j in range(nk):
    k = klum_thi_klai[j]
    if(maksutmai[k]):
        z = k # ถ้ามากสุดก็ได้คำตอบเป็นกลุ่มนี้
        break
print(u'5 กลุ่มใกล้สุด %s\nคำตอบ %d'%(klum_thi_klai[:5],z))

# สุ่มจุดในบริเวณนี้มาสัก ๑๕ จุด แล้วหากลุ่มของแต่ละจุด

n = 15
x = np.random.uniform(tamnaeng[:,0].min(),tamnaeng[:,0].max(),n)
y = np.random.uniform(tamnaeng[:,1].min(),tamnaeng[:,1].max(),n)
X = np.stack([x,y],1)

# ป้อนค่าตำแหน่งหลายจุดเป็นอาเรย์คำนวณพร้อมกันได้
nk = 5
raya2 = ((X[None]-tamnaeng[:,None])**2).sum(2)
klum_thi_klai = klum[raya2.argsort(0)]
n_nai_klum = np.stack([(klum_thi_klai[:nk]==k).sum(0) for k in range(n_klum)])
mi_maksut = n_nai_klum.max(0)
maksutmai = (n_nai_klum==mi_maksut)
z = np.empty(n,dtype=int)
for i in range(n):
    for j in range(nk):
        k = klum_thi_klai[j,i]
        if(maksutmai[k,i]):
            z[i] = k
            break

# นำมาวาดภาพ โดยจุดที่ทำนายกลุ่มให้เป็นสี่เหลี่ยมขอบแดง ถูกแบ่งกลุ่มตรงกับจุดกลมที่อยู่ใก
plt.figure(figsize=[6,6])
plt.gca(aspect=1)
plt.scatter(tamnaeng[:,0],tamnaeng[:,1],c=klum,edgecolor='k',cmap='viridis')
plt.scatter(x,y,60,c=z,marker='s',edgecolor='r')
plt.show()

n_klum = 4
np.random.seed(10)
tamnaeng,klum = datasets.make_blobs(n_samples=200,n_features=2,centers=12,cluster_std=1.4)
klum %= n_klum
plt.figure(figsize=[8,8])
plt.gca(aspect=1)
plt.scatter(tamnaeng[:,0],tamnaeng[:,1],c=klum,edgecolor='k',cmap='viridis')
nk = 5 # จำนวนเพื่อนบ้าน
pb = Phueanban(nk)
pb.rianru(tamnaeng,klum)

# จุดข้อมูลทดสอบ
n = 20
x = np.random.uniform(tamnaeng[:,0].min(),tamnaeng[:,0].max(),n)
y = np.random.uniform(tamnaeng[:,1].min(),tamnaeng[:,1].max(),n)
X = np.stack([x,y],1)
z = pb.thamnai(X)
plt.scatter(x,y,60,c=z,marker='s',edgecolor='#ff6666')

# ระบายสีพื้น
nmesh = 200
mx,my = np.meshgrid(np.linspace(-15,15,nmesh),np.linspace(-15,15,nmesh))
mX = np.stack([mx.ravel(),my.ravel()],1)
mz = pb.thamnai(mX).reshape(nmesh,nmesh)
plt.pcolormesh(mx,my,mz,alpha=0.1,cmap='viridis')
plt.contour(mx,my,mz,alpha=0.3,colors='#6666ff')
plt.show()