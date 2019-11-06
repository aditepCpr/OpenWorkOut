import cv2
import os
from sys import platform
from face.draw_boundary import detect_face
# from body.draw_body import detect_body
from tkinter import *
import gui

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('/usr/local/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


class OpenWorkpout:
    print('OpenWorkout Ok...')

    def __init__(self, filename, nameEx, nameEx2):
        self.filename = filename
        self.nameEx = nameEx
        self.nameEx2 = nameEx2

    def _OpenCVpose(self):
        print(type(self.filename), self.filename)
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "body/models/"

        # impost model face
        faceCascade = cv2.CascadeClassifier('face/model/haarcascade_frontalface_default.xml')

        # impost classifier
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("face/model/classifier.xml")

        try:
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()

            # Process Image
            datum = op.Datum()
            # imageToProcess = cv2.VideoCapture(0)
            # model
            try:
                file_model = open('model/MLPClassifier.pkl', 'rb')
                self.model = pickle.load(file_model)
                file_model.close()
            except IOError as e:
                print(e)

            imageToProcess = cv2.VideoCapture(self.filename)
            img_id = 1
            path = [os.path.join('dataSet/' + self.nameEx, f) for f in os.listdir('dataSet/' + self.nameEx)]
            if len(path) >= 1:
                if self.nameEx != 'cam':
                    img_id = len(path)
            while (imageToProcess.isOpened()):
                # Read Video
                ret, frame = imageToProcess.read()
                # face
                # f1 = detect_face(ret, frame, faceCascade, img_id, clf)

                # datum.openpose input data

                datum.cvInputData = frame
                opWrapper.emplaceAndPop([datum])
                bodyKeypoints = datum.poseKeypoints
                try:
                    f1 = detect_body(frame, bodyKeypoints, img_id, self.nameEx, self.nameEx2, self.model)
                except Exception as e:
                    print(e)

                # print("Body keypoints: \n" + str(datum.poseKeypoints))

                cv2.imshow('OpenWorkOut', f1)
                # cv2.imshow('OpenWorkOut')
                img_id += 1
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break


        except Exception as e:
            print(e)
            imageToProcess.release()
            cv2.destroyAllWindows()
            # sys.exit(-1)

        imageToProcess.release()
        cv2.destroyAllWindows()


# !/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from body.KeyPoints import KeyPoints
import CreateJson as Cjson
import Predict_Live


def draw_boundary(img, bodyKeypoints, kp,z,Z):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # แขนขวา

    if (kp.getRShoulder1() > 0 ):
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
    return img


def detect_body(img, bodyKeypoints, img_id, nameEx, nameEx2, model):
    # print('detect_body Ok...')

    kp = KeyPoints(bodyKeypoints)

    mlpc_ = Mlpc(kp.getAllKeypoints(), nameEx2, model)
    # 0 = squat : 1 =  curl :  2  =  pushup  : 3  =  dumbbellShoulderPress : 4  =  deadlift
    img = draw_boundary(img, bodyKeypoints, kp,mlpc_.mlpc(),1)
    # mlpc_.mlpc()
    print(mlpc_.mlpc())

    # create Json file
    # Cjson.CreateJson(kp, img_id, nameEx)
    return img


import pickle
import numpy as np


def stackData_PredictEx(X):
    x = xxEx(X)
    y = yyEx(X)
    X_ = np.stack((x, y), axis=1)
    return X_


def xxEx(xy):
    X = []
    for x in xy:
        npx = np.array(x)
        X.append(npx[0])
    return X


def yyEx(xy):
    Y = []
    for y in xy:
        npx = np.array(y)
        Y.append(npx[1])
    return Y


class Mlpc:
    mz = []
    mx = []
    my = []
    mX = []
    mz_ = []
    target_names = []

    def __init__(self, X, nameEx, model):
        self.X = X
        self.model = model
        self.nameEx = nameEx
        self.name = 'Multi-layer Perceptron classifier'
        self.mz_ = self.mz_
        self.fileName = 'MLPClassifier'
        self.target_names = np.array(['Dumbbell ShoulderPress', 'unknown'], dtype='<U10')

    def mlpc(self):
        X_ = stackData_PredictEx(self.X)
        self.mz_ = self.model.predict(X_)
        return self.mz_


# test
if __name__ == '__main__':
    opw = OpenWorkpout(0, 'cam', 'Dumbbell Shoulder Press')
    opw._OpenCVpose()

#
