from ReadData import CreateData as cd
import numpy as np
from sklearn.model_selection import train_test_split
class StackData:
    def __init__(self, path):
        self.path = path

    def stackData_Train(self):
        idc = 0
        nxy, z = cd.allpath(self.path, idc)
        x = cd.xx(nxy)
        y = cd.yy(nxy)
        z = cd.cen_z(z)
        X = np.stack((x, y), axis=1)
        z = np.array(z)
        X = (X - X.mean(0)) / X.std(0)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
        return X_train, X_test, z_train, z_test


    def stackData_Predict(self):
        idc = 0
        nxy, z = cd.allpath(self.path, idc)
        x = cd.xx(nxy)
        y = cd.yy(nxy)
        z = cd.cen_z(z)
        X = np.stack((x, y), axis=1)
        z = np.array(z)
        X = (X - X.mean(0)) / X.std(0)
        return X, z

class DataModel:
    def __init__(self):
        print('dataModel')
        self.squat = cd("dataSet/Squat")
        self.curl = cd("dataSet/Barbell Curl")
        self.pushup = cd('dataSet/Push Ups')
        self.dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
        self.deadlift = cd('dataSet/Deadlift')
        self.cam = cd('dataSet/cam')
        self.target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')

    def getSquat(self):
        return self.squat

    def getCurl(self):
        return self.curl

    def getPushup(self):
        return self.pushup

    def getDumbbellShoulderPress(self):
        return self.dumbbellShoulderPress

    def getDeadlift(self):
        return self.deadlift

    def getCam(self):
        return self.cam

    def getTargetNames(self):
        return self.target_names
