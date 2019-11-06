from StackData import stackData_PredictEx
class Predict_Live:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []

    def __init__(self, X, nameEx,model):
        self.X = X
        self.model = model
        self.nameEx = nameEx
        self.name = 'Multi-layer Perceptron classifier'
        self.mz_ = self.mz_
        self.fileName = 'MLPClassifier'
    def predictLive(self):
        X_ = stackData_PredictEx(self.X)
        self.mz_ = self.model.predict(X_)
        return self.mz_

def predict_Data(X, model):
    mz_ = model.predict(X)
    return mz_

