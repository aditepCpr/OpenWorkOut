import numpy as np
class KeyPoints:

    def __init__(self,bodyKeypoints):
        self.bkp = bodyKeypoints
        self.x = np.array(self.bkp)

        # คอ
        self.Neck1 = self.x[0][1][0]
        self.Neck2 = self.x[0][1][1]

        # ไหล่ขวา
        self.RShoulder1 = self.x[0][2][0]
        self.RShoulder2 = self.x[0][2][1]

        # แขนบนขวา
        self.RElbow1 = self.x[0][3][0]
        self.RElbow2 = self.x[0][3][1]

        # แขนล่างขวา
        self.RWrist1 = self.x[0][4][0]
        self.RWrist2 = self.x[0][4][1]

        # ไหล่ซ้าย
        self.LShoulder1 = self.x[0][5][0]
        self.LShoulder2 = self.x[0][5][1]

        # แขนบนซ้าย
        self.LElbow1 = self.x[0][6][0]
        self.LElbow2 = self.x[0][6][1]

        # แขนล่างซ้าย

        self.LWrist1 = self.x[0][7][0]
        self.LWrist2 = self.x[0][7][1]

        # ลำตัว

        self.MidHip1 = self.x[0][8][0]
        self.MidHip2 = self.x[0][8][1]

        # สะโพกขวา
        self.RHip1 = self.x[0][9][0]
        self.RHip2 = self.x[0][9][1]

        # ขาบนขวา
        self.RKnee1 = self.x[0][10][0]
        self.RKnee2 = self.x[0][10][1]

        # ขาล่างขวา
        self.RAnkle1 = self.x[0][11][0]
        self.RAnkle2 = self.x[0][11][1]

        # สะโพกซ้าย
        self.LHip1 = self.x[0][12][0]
        self.LHip2 = self.x[0][12][1]

        # ขาบนซ้าย
        self.LKnee1 = self.x[0][13][0]
        self.LKnee2 = self.x[0][13][1]

        # ขาล่างซ้าย
        self.LAnkle1 = self.x[0][14][0]
        self.LAnkle2 = self.x[0][14][1]




    def getNeck1(self):
        return self.Neck1

    def getNeck2(self):
        return self.Neck2

    def getRShoulder1(self):
        return self.RShoulder1

    def getRShoulder2(self):
        return self.RShoulder2

    def getRElbow1(self):
        return self.RElbow1

    def getRElbow2(self):
        return self.RElbow2

    def getRWrist1(self):
        return self.RWrist1

    def getRWrist2(self):
        return self.RWrist2

    def getLShoulder1(self):
        return self.LShoulder1

    def getLShoulder2(self):
        return self.LShoulder2

    def getLElbow1(self):
        return self.LElbow1

    def getLElbow2(self):
        return self.LElbow2

    def getLWrist1(self):
        return self.LWrist1

    def getLWrist2(self):
        return self.LWrist2

    def getMidHip1(self):
        return self.MidHip1

    def getMidHip2(self):
        return self.MidHip2

    def getRHip1(self):
        return self.RHip1

    def getRHip2(self):
        return self.RHip2

    def getRKnee1(self):
        return self.RKnee1

    def getRKnee2(self):
        return self.RKnee2

    def getLHip1(self):
        return self.LHip1

    def getLHip2(self):
        return self.LHip2

    def getLKnee1(self):
        return self.LKnee1

    def getLKnee2(self):
        return self.LKnee2

    def getLAnkle1(self):
        return self.LAnkle1

    def getLAnkle2(self):
        return self.LAnkle2

    def getRAnkle1(self):
        return self.RAnkle1

    def getRAnkle2(self):
        return self.RAnkle2

    def getBodyKeypoints(self):
        return self.bkp