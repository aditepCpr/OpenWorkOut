from ReadData import CreateData as cd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler as Sta
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
        # X = (X - X.mean(0)) / X.std(0)
        # sta = Sta()
        # sta.fit(X)
        # X = sta.transform(X)
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
        # sta = Sta()
        # sta.fit(X)
        # X = sta.transform(X)
        # X = (X - X.mean(0)) / X.std(0)
        return X, z



class DataModel:
    # nameEx = []
    # target_namesEx = []
    def __init__(self):
        print('dataModel')
        self.squat = cd("dataSet/Squat")
        self.curl = cd("dataSet/Barbell Curl")
        self.pushup = cd('dataSet/Push Ups')
        self.dumbbellShoulderPress = cd('dataSet/Dumbbell Shoulder Press')
        self.deadlift = cd('dataSet/Deadlift')
        self.cam = cd('dataSet/cam')
        self.unknown = cd('dataSet/unknown')
        self.target_names = np.array(['squat', 'curl', 'pushup', 'dumbbellShoulderPress', 'deadlift'], dtype='<U10')
        self.path = []

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

    def getUnknown(self):
        return self.unknown

    def getTargetNames(self):
        return self.target_names

    def DataModelEx(self,nameEX):
        if nameEX == 'Push Ups':
            self.path = [self.pushup, self.unknown]
            self.target_names = np.array(['Push Ups', 'unknown'], dtype='<U10')
        elif nameEX == 'Squat':
            self.path = [self.squat, self.unknown]
            self.target_names = np.array(['Squat', 'unknown'], dtype='<U10')
        elif nameEX == 'Barbell Curl':
            self.path = [self.curl, self.unknown]
            self.target_names = np.array(['Barbell Curl', 'unknown'], dtype='<U10')
        elif nameEX == 'Dumbbell Shoulder Press':
            self.path = [self.dumbbellShoulderPress, self.unknown]
            self.target_names = np.array(['Dumbbell ShoulderPress', 'unknown'], dtype='<U10')
        elif nameEX == 'Deadlift':
            self.path = [self.deadlift, self.unknown]
            self.target_names = np.array(['Dead lift', 'unknown'], dtype='<U10')
        return self.path,self.target_names

def load_Data(fileName):
    try:
        file_model = open('model/'+fileName + '.pkl', 'rb')
        model = pickle.load(file_model)
        file_model.close()
    except IOError as e:
        print(e)
    return model

def load_DataEx(fileName,nameEx):
    try:
        file_model = open('model/'+nameEx+'/'+fileName + '.pkl', 'rb')
        model = pickle.load(file_model)
        file_model.close()
    except IOError as e:
        print(e)
    return model
