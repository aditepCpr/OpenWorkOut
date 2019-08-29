from createData import CreateData as cd
import numpy as np
import OpenWorkout2  as Owk2
squat = cd("dataSet/squat")
curl = cd("dataSet/curl")
pushup = cd('dataSet/pushup')
path = [squat,curl,pushup]
idc = 0


nxy,z = cd.allpath(path,idc)
xxx = cd.xx(nxy)
yyy = cd.yy(nxy)
z = cd.cen_z(z)
supperxy = np.stack((xxx,yyy),axis=1)
print(xxx,yyy)

owk = Owk2.OpenWorkpout2('/home/aditep/soflware/OpenWorkOut/vdo/BarbellCurl/BarbellCurl1.mp4')
owk._OpenCVpose()

