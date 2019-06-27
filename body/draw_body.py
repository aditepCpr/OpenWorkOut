import cv2
import numpy as np


def draw_boundary(img, bodyKeypoints, color):
    print('draw_boby', bodyKeypoints)
    x = np.array(bodyKeypoints)
    print('shape', x.shape)
    print('ndim', x.ndim)
    print('size ndarray', x.size)

    # คอ
    Neck1 = x[0][1][0]
    Neck2 = x[0][1][1]

    # ไหล่ขวา
    RShoulder1 = x[0][2][0]
    RShoulder2 = x[0][2][1]
    # แขนบนขวา
    RElbow1 = x[0][3][0]
    RElbow2 = x[0][3][1]
    # แขนล่างขวา
    RWrist1 = x[0][4][0]
    RWrist2 = x[0][4][1]

    # ไหล่ซ้าย
    LShoulder1 = x[0][5][0]
    LShoulder2 = x[0][5][1]
    # แขนบนซ้าย
    LElbow1 = x[0][6][0]
    LElbow2 = x[0][6][1]
    # แขนล่างซ้าย
    LWrist1 = x[0][7][0]
    LWrist2 = x[0][7][1]

    # ลำตัว

    MidHip1 = x[0][8][0]
    MidHip2 = x[0][8][1]

    # สะโพกขวา
    RHip1 = x[0][9][0]
    RHip2 = x[0][9][1]

    # ขาบนขวา
    RKnee1 = x[0][10][0]
    RKnee2 = x[0][10][1]

    # ขาล่างขวา
    RAnkle1 = x[0][11][0]
    RAnkle2 = x[0][11][1]

    # สะโพกซ้าย
    LHip1 = x[0][12][0]
    LHip2 = x[0][12][1]

    # ขาบนซ้าย
    LKnee1 = x[0][13][0]
    LKnee2 = x[0][13][1]

    # ขาล่างซ้าย
    LAnkle1 = x[0][14][0]
    LAnkle2 = x[0][14][1]
    print(LHip1,LHip2)
    # draw

    # แขนขวา
    if (RShoulder1 != 0):
        cv2.line(img, (Neck1, Neck2), (RShoulder1, RShoulder2), (0, 0, 255), 10)
    if (RElbow1 != 0):
        cv2.line(img, (RShoulder1, RShoulder2), (RElbow1, RElbow2), (0, 0, 255), 10)
    if (RWrist1 != 0):
        cv2.line(img, (RElbow1, RElbow2), (RWrist1, RWrist2), (0, 0, 255), 10)

    # แขนซ้าย
    if (LShoulder1 != 0):
        cv2.line(img, (Neck1, Neck2), (LShoulder1, LShoulder2), (0, 0, 255), 10)
    if (LElbow1 != 0):
        cv2.line(img, (LShoulder1, LShoulder2), (LElbow1, LElbow2), (0, 0, 255), 10)
    if (LWrist1 != 0):
        cv2.line(img, (LElbow1, LElbow2), (LWrist1, LWrist2), (0, 0, 255), 10)

    # ลำตัว
    if (MidHip1 != 0):
        cv2.line(img, (Neck1, Neck2), (MidHip1, MidHip2), (0, 0, 255), 10)

    # ขาขวา
    if (RHip1 != 0):
        cv2.line(img, (MidHip1, MidHip2), (RHip1, RHip2), (0, 0, 255), 10)
    if (RKnee1 != 0):
        cv2.line(img, (RHip1, RHip2), (RKnee1, RKnee2), (0, 0, 255), 10)
    if (RAnkle1 != 0):
        cv2.line(img, (RKnee1, RKnee2), (RAnkle1, RAnkle2), (0, 0, 255), 10)

    # ขาซ้าย
    if (LHip1 != 0):
        cv2.line(img, (MidHip1, MidHip2), (LHip1, LHip2), (0, 0, 255), 10)
    if (LKnee1 != 0):
        cv2.line(img, (LHip1, LHip2), (LKnee1, LKnee2), (0, 0, 255), 10)
    if (LAnkle1 != 0):
        cv2.line(img, (LKnee1, LKnee2), (LAnkle1, LAnkle2), (0, 0, 255), 10)



def detect_body(img, bodyKeypoints):
    img = draw_boundary(img, bodyKeypoints, (0, 0, 255))
