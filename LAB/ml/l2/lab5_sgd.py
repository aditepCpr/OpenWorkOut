##  stochastic gradient descent
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ha_entropy(z, h):
    return -(z * np.log(h) + (1 - z) * np.log(1 - h))


# คำตอบค่า AND

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
z = np.array([0, 0, 0, 1])

w = np.array([0, 0.])  # พารามิเตอร์ตั้งต้น
b = 0
n = len(z)  # จำนวนข้อมูล
eta = 0.8  # อัตราการเรียนรู้
thamsam = 250
entropy = []
for o in range(thamsam):
    for i in range(n):
        ai = np.dot(X[i], w) + b
        hi = sigmoid(ai)
        gai = hi - z[i]
        gwi = gai * X[i]
        gbi = gai
        w -= eta * gwi  # ปรับค่าพารามิเตอร์
        b -= eta * gbi
        J = ha_entropy(z[i], hi)  # คำนวณค่าเสียหายเก็บไว้
        entropy.append(J)

import matplotlib.pyplot as plt

# วาดแสดงการแบ่งพื้นที่
mx, my = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
mX = np.array([mx.ravel(), my.ravel()]).T
mh = np.dot(mX, w) + b
mz = (mh >= 0).astype(int).reshape(200, -1)
plt.gca(aspect=1)
plt.contourf(mx, my, mz, cmap='spring')
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='b', marker='D', cmap='gray')
plt.show()

mz = sigmoid(mh).reshape(200, -1)
plt.gca(aspect=1)
plt.contourf(mx, my, mz, cmap='spring')
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='b', marker='D', cmap='gray')
plt.show()

plt.plot(entropy, 'r')
plt.xlabel(u'จำนวนรอบ', family='tahoma', size=14)
plt.ylabel(u'ค่าเสียหาย', family='tahoma', size=14)
plt.show()

## คำนวณค่าเสียหายรวมของทุกตัวแล้วจึงนำมาเฉลี่ยแล้วค่อยปรับพารามิเตอร์ทีเดียว
w = np.array([0, 0.])
b = 0
eta = 0.8
thamsam = 1000
entropy = []
for o in range(thamsam):
    dw = 0
    db = 0
    J = 0
    for i in range(n):
        ai = np.dot(X[i], w) + b
        hi = sigmoid(ai)
        gai = hi - z[i]
        gwi = gai * X[i]
        gbi = gai
        dw -= eta * gwi
        db -= eta * gbi
        J += ha_entropy(z[i], hi)
    w += dw / n
    b += db / n
    entropy.append(J)

plt.figure()
plt.plot(entropy, 'r')
plt.xlabel(u'จำนวนรอบ', family='tahoma', size=14)
plt.ylabel(u'ค่าเสียหาย', family='tahoma', size=14)
plt.show()
