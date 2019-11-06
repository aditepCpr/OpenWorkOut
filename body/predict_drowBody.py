#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from body.KeyPoints import KeyPoints
import CreateJson as Cjson
import Predict_Live


def draw_boundary(img, bodyKeypoints, kp, z, Z,nameEx2):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # แขนขวา

    if (kp.getRShoulder1() > 0):
        if z[0] == Z and z[1] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getNeck1(), kp.getNeck2()), (kp.getRShoulder1(), kp.getRShoulder2()),
                 color, 10)
    if (kp.getRElbow1() > 0):
        if z[1] == Z and z[2] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getRShoulder1(), kp.getRShoulder2()), (kp.getRElbow1(), kp.getRElbow2()), color, 10)
    if (kp.getRWrist1() > 0):
        if z[2] == Z and z[3] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getRElbow1(), kp.getRElbow2()), (kp.getRWrist1(), kp.getRWrist2()), color, 10)

    # แขนซ้าย
    if (kp.getLShoulder1() > 0):
        if z[3] == Z and z[4] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getNeck1(), kp.getNeck2()), (kp.getLShoulder1(), kp.getLShoulder2()), color, 10)
    if (kp.getLElbow1() > 0):
        if z[4] == Z and z[5] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getLShoulder1(), kp.getLShoulder2()), (kp.getLElbow1(), kp.getLElbow2()), color, 10)
    if (kp.getLWrist1() > 0):
        if z[5] == Z and z[6] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getLElbow1(), kp.getLElbow2()), (kp.getLWrist1(), kp.getLWrist2()), color, 10)

    # ลำตัว
    if (kp.getMidHip1() > 0):
        if z[6] == Z and z[7] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getNeck1(), kp.getNeck2()), (kp.getMidHip1(), kp.getMidHip2()), color, 10)

    # ขาขวา
    if (kp.getRHip1() > 0):
        if z[7] == Z and z[8] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getMidHip1(), kp.getMidHip2()), (kp.getRHip1(), kp.getRHip2()), color, 10)
    if (kp.getRKnee1() > 0):
        if z[8] == Z and z[9] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getRHip1(), kp.getRHip2()), (kp.getRKnee1(), kp.getRKnee2()), color, 10)
    if (kp.getRAnkle1() > 0):
        if z[9] == Z and z[10] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getRKnee1(), kp.getRKnee2()), (kp.getRAnkle1(), kp.getRAnkle2()), color, 10)

    # ขาซ้าย
    if (kp.getLHip1() > 0):
        if z[10] == Z and z[11] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getMidHip1(), kp.getMidHip2()), (kp.getLHip1(), kp.getLHip2()), color, 10)
    if (kp.getLKnee1() > 0):
        if z[11] == Z and z[12] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        cv2.line(img, (kp.getLHip1(), kp.getLHip2()), (kp.getLKnee1(), kp.getLKnee2()), color, 10)
    if (kp.getLAnkle1() > 0):
        if z[12] == Z and z[13] == Z:
            color = (206, 219, 5)
        else:
            color = (0, 0, 255)
        img = cv2.line(img, (kp.getLKnee1(), kp.getLKnee2()), (kp.getLAnkle1(), kp.getLAnkle2()), color, 10)
    cv2.putText(img, str(nameEx2), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img


def detect_body(img, bodyKeypoints, img_id, nameEx, nameEx2,model):
    kp = KeyPoints(bodyKeypoints)
    predictLive = Predict_Live.Predict_Live(kp.getAllKeypoints(), nameEx2, model)
    # 0 = squat : 1 =  curl :  2  =  pushup  : 3  =  dumbbellShoulderPress : 4  =  deadlift
    img = draw_boundary(img, bodyKeypoints, kp, predictLive.predictLive(), 0,nameEx2)
    print(predictLive.predictLive())
    # create Json file
    return img
