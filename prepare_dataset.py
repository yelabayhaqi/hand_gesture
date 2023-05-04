import cv2
import glob
import math
import os

from cvzone.HandTrackingModule import HandDetector
import numpy as np

detector = HandDetector(maxHands=1)

offset = 0
imgSize = 400

def convert_img(dir, num):
    i = 1
    os.mkdir("dataset_fix/"+num)
    for images in glob.glob(dir + "*.jpeg"):
        img = cv2.imread(images)
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)
            # imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))

                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))

                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            #cv2.imshow("Image_crop", imgCrop)
            cv2.imshow("Image_white", imgWhite)
            cv2.imwrite("dataset_fix/"+num+"/hand_"+num+"_"+str(i)+".jpg", imgWhite)
        cv2.imshow("img_ori", img)
        cv2.waitKey(1)
        i = i+1

for i in range(10):
    idx = str(i)
    convert_img("dataset/"+idx+"/", idx)
    print(idx)
for x in range(ord('a'), ord('z') + 1):
    idx = str(chr(x))
    convert_img("dataset/"+idx+"/", idx)
    print(idx)