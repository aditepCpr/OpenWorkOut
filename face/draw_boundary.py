#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import cv2
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    print('draw_boundary')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)

    coords = []

    for (x, y, w, h) in features:
        # print('face',features)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # id กับ ค่าความผิดพลาด
        id, con = clf.predict(gray[y:y + h, x:x + w])

        if con <= 80:
            cv2.putText(img, 'TOP', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        else:
            cv2.putText(img, 'unKnow', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

        if (con < 100):
            con = "{0}%".format(round(100 - con))
        else:
            con = "{0}%".format(round(100 - con))
        img = cv2.putText(img, str(con), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # print(str(con))

        coords = [x, y, w, h]
    return img, coords

def detect_face(img, faceCascade, img_id, clf):
    print('detect_face')
    img, coords = draw_boundary(img, faceCascade, 1.1, 10, (0, 0, 255), clf)

    # ตรวจจับเฉพาะหน้า
    if len(coords) == 4:
        id = 1
        result = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
    return img

