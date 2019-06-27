from body.KeyPoints import *
import json as js
import numpy as np
class CreateJson():

    def __init__(self,kp):
        self.kp = kp

        # print(KeyPoints.getBodyKeypoints(self.kp))
        KeyPoints.getMidHip1(kp)
        KeyPoints.getMidHip1(kp)
        BodyKey = np.array([[KeyPoints.getNeck1(kp),KeyPoints.getNeck2(kp)],[KeyPoints.getRShoulder1(kp),KeyPoints.getRShoulder2(kp)],
                            [KeyPoints.getRElbow1(kp),KeyPoints.getRElbow1(kp)],[KeyPoints.getRWrist1(kp),KeyPoints.getRWrist2(kp)],
                            [KeyPoints.getLShoulder1(kp),KeyPoints.getLShoulder2(kp)],[KeyPoints.getLElbow1(kp),KeyPoints.getLElbow2(kp)],
                            [KeyPoints.getLWrist1(kp),KeyPoints.getLWrist2(kp)],[KeyPoints.getMidHip1(kp),KeyPoints.getMidHip2(kp)],
                            [KeyPoints.getRHip1(kp),KeyPoints.getRHip2(kp)],[KeyPoints.getRKnee1(kp),KeyPoints.getRKnee2(kp)],
                            [KeyPoints.getRAnkle1(kp),KeyPoints.getRAnkle2(kp)],[KeyPoints.getLHip1(kp),KeyPoints.getLHip2(kp)],
                            [KeyPoints.getLKnee1(kp),KeyPoints.getLKnee2(kp)],[KeyPoints.getLAnkle1(kp),KeyPoints.getLAnkle2(kp)]])
        print(BodyKey)
    #
    # try:
    #     # json
    #     keypose = str()
    #     key = open('keypose.json', 'w')
    #     # key = open('keypose.json','a')
    #     js.dump(keypose, key, indent=4)
    # except  IOError as e:
    #     print(e)
    #
    #

