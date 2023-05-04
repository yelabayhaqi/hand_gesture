import cv2
import glob
import math

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/model_black/keras_model.h5", "Model/model_black/labels.txt")

offset = 5
imgSize = 400

labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
          "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
          "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
          "u", "v", "w", "x", "y", "z"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.putText(imgOutput, labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
        cv2.rectangle(imgOutput, (x-offset,y-offset),
                      (x+w+offset, y+h+offset), (255,0,255), 4)

        cv2.imshow("Image_crop", imgCrop)
        cv2.imshow("Image_white", imgWhite)

    cv2.imshow("cam", imgOutput)
    cv2.waitKey(100)