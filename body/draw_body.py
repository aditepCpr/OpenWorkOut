#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import cv2
import numpy as np
from body.KeyPoints import KeyPoints
import LAB.knn.k_sklearn.labsklean.lab01 as lab01
import CreateJson as Cjson

def draw_boundary(img, bodyKeypoints, color,kp):

    # แขนขวา

    if (kp.getRShoulder1() != 0):
        cv2.line(img, (kp.getNeck1(), kp.getNeck2()), (kp.getRShoulder1(), kp.getRShoulder2()),
                 (0, 0, 255), 10)
    if (kp.getRElbow1() != 0):
        cv2.line(img, (kp.getRShoulder1(), kp.getRShoulder2()), (kp.getRElbow1(), kp.getRElbow2()),color, 10)
    if (kp.getRWrist1() != 0):
        cv2.line(img, (kp.getRElbow1(), kp.getRElbow2()), (kp.getRWrist1(), kp.getRWrist2()),color, 10)

    # แขนซ้าย
    if (kp.getLShoulder1() != 0):
        cv2.line(img, (kp.getNeck1(), kp.getNeck2()), (kp.getLShoulder1(), kp.getLShoulder2()), color, 10)
    if (kp.getLElbow1() != 0):
        cv2.line(img, (kp.getLShoulder1(), kp.getLShoulder2()), (kp.getLElbow1(), kp.getLElbow2()),color, 10)
    if (kp.getLWrist1() != 0):
        cv2.line(img, (kp.getLElbow1(), kp.getLElbow2()), (kp.getLWrist1(), kp.getLWrist2()), color, 10)

    # ลำตัว
    if (kp.getMidHip1() != 0):
        cv2.line(img, (kp.getNeck1(), kp.getNeck2()), (kp.getMidHip1(), kp.getMidHip2()), color, 10)

    # ขาขวา
    if (kp.getRHip1() != 0):
        cv2.line(img, (kp.getMidHip1(), kp.getMidHip2()), (kp.getRHip1(), kp.getRHip2()), color, 10)
    if (kp.getRKnee1() != 0):
        cv2.line(img, (kp.getRHip1(), kp.getRHip2()), (kp.getRKnee1(), kp.getRKnee2()), color, 10)
    if (kp.getRAnkle1() != 0):
        cv2.line(img, (kp.getRKnee1(), kp.getRKnee2()), (kp.getRAnkle1(), kp.getRAnkle2()), color, 10)

    # ขาซ้าย
    if (kp.getLHip1() != 0):
        cv2.line(img, (kp.getMidHip1(), kp.getMidHip2()), (kp.getLHip1(), kp.getLHip2()), color, 10)
    if (kp.getLKnee1() != 0):
        cv2.line(img, (kp.getLHip1(), kp.getLHip2()), (kp.getLKnee1(), kp.getLKnee2()), color, 10)
    if (kp.getLAnkle1() != 0):
        cv2.line(img, (kp.getLKnee1(), kp.getLKnee2()), (kp.getLAnkle1(), kp.getLAnkle2()), color, 10)



def detect_body(img, bodyKeypoints,img_id):

    kp = KeyPoints(bodyKeypoints)

    color = (0,0,255)
    img = draw_boundary(img, bodyKeypoints, color,kp)

    # create Json file
    # Cjson.CreateJson(kp,img_id)