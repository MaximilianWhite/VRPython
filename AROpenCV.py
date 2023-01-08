import requests
import cv2
import numpy as np


cap = cv2.VideoCapture(1)

imgTarget = cv2.imread('zvezd.png')
myVid = cv2.VideoCapture('video.mp4')

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

orb = cv2.ORB_create(nfeatures=200)

kp1, des1 = orb.detectAndCompute(imgTarget, None)

while True:
    success, imgWebcam = cap.read()

    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

    if len(good) > 20:
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print(matrix)

        pts = np.float32([[0,0], [0,hT], [wT, hT], [wT,0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255,0,255),3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], (255,255,255))
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInv)
        imgAug = cv2.bitwise_or(imgAug, imgAug)

        cv2.imshow('img2', img2)
        # cv2.imshow('maskNew', imgAug)
        # cv2.imshow('imgWarp', imgWarp)

        cv2.imshow('imgFeatures', imgFeatures)
    cv2.waitkey(0)
