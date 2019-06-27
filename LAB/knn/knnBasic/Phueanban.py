import numpy as np
class Phueanban:
    def __init__(self,nk):
        self.nk = nk # จำนวนเพื่อนบ้านที่จะพิจารณา

    def rianru(self,X,z):
        self.X = X # เก็บข้อมูลตำแหน่ง
        self.z = z # เก็บข้อมูลการแบ่งกลุ่ม
        self.n_klum = z.max()+1 # จำนวนกลุ่ม

    def thamnai(self,X):
        n = len(X) # จำนวนข้อมูลที่จะคำนวณหา
        raya2 = ((X[None]-self.X[:,None])**2).sum(2)
        klum_thi_klai = self.z[raya2.argsort(0)]
        n_nai_klum = np.stack([(klum_thi_klai[:self.nk]==k).sum(0) for k in range(self.n_klum)])
        mi_maksut = n_nai_klum.max(0)
        maksutmai = (n_nai_klum==mi_maksut)
        z = np.empty(n,dtype=int)
        for i in range(n):
            for j in range(self.nk):
                k = klum_thi_klai[j,i]
                if(maksutmai[k,i]):
                    z[i] = k
                    break
        return z

