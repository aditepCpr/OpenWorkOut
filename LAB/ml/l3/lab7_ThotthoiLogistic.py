import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ha_entropy(z, h):
    return -(z * np.log(h) + (1 - z) * np.log(1 - h)).mean()


class ThotthoiLogistic:
    def __init__(self, eta):
        self.eta = eta

    def rianru(self, X, z, n_thamsam):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.entropy = []
        self.khanaen = []
        for i in range(n_thamsam):
            a = self.ha_a(X)
            h = sigmoid(a)
            J = ha_entropy(z, h)
            ga = (h - z) / len(z)
            self.w -= self.eta * np.dot(ga, X)
            self.b -= self.eta * ga.sum()
            self.entropy.append(J)
            knanaen = ((a >= 0) == z).mean()
            self.khanaen.append(knanaen)

    def thamnai(self, X):
        return (self.ha_a(X) >= 0).astype(int)

    def ha_a(self, X):
        return np.dot(X, self.w) + self.b


if __name__ == '__main__':
    from glob import glob

    d = 25
    X1 = np.array([plt.imread(x) for x in glob('ruprang-raisi-25x25x1000x5/0/*.png')]).reshape(-1, 625)
    X2 = np.array([plt.imread(x) for x in glob('ruprang-raisi-25x25x1000x5/1/*.png')]).reshape(-1, 625)
    X = np.vstack([X1[:900], X2[:900]])  # คัดเฉพาะ 900 รูปแรกของแต่ละแบบ นำมารวมกัน
    z = np.arange(2).repeat(900)  # คำตอบ เลข 0 และ 1
    tl = ThotthoiLogistic(eta=0.01)  # สร้างออบเจ็กต์ของคลาสการถดถอยโลจิสติก
    tl.rianru(X, z, n_thamsam=1000)  # ทำการเรียนรู้
    plt.subplot(211, xticks=[])
    plt.plot(tl.entropy, 'm')
    plt.ylabel(u'เอนโทรปี', family='Tahoma')
    plt.subplot(212)
    plt.plot(tl.khanaen, 'm')
    plt.ylabel(u'คะแนน', family='Tahoma')
    plt.xlabel(u'จำนวนรอบ', family='Tahoma')
    plt.show()

    # นำข้อมูล 100 ตัวที่เหลือมาลองทำนายผล แล้วเทียบกับคำตอบจริง
    Xo = np.vstack([X1[900:], X2[900:]])
    zo = np.arange(2).repeat(100)
    print((tl.thamnai(Xo) == zo).mean())  # ได้ 0.92
