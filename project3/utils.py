import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
plt.close('all')


def PointsTransform(points, M):
    points_tr = np.hstack([points, np.ones([points.shape[0], 1])]) @ M.T
    points_tr = points_tr[:, :2] / points_tr[:, 2:]

    return points_tr



def SiftKeypoints(image):
    sift = cv2.SIFT_create()
    keypoints, features = sift.detectAndCompute(image, None)

    return keypoints, features


def MatchPoints(features1, features2, threshold=0.4):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(features1, features2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            goodMatch.append(m)

    if len(goodMatch) < 10:
        return MatchPoints(features1, features2, threshold + 0.1)
    else:
        return goodMatch


def ComputeH(pointsMove, pointsFixed):
    H, status = cv2.findHomography(pointsMove, pointsFixed, cv2.RANSAC, 4)

    return H

def AffineMatrix(imageFixed, imageMove):
    # sift feature extract
    keypointsFixed, featuresFixed = SiftKeypoints(imageFixed)
    keypointsMove, featuresMove = SiftKeypoints(imageMove)

    # match points
    goodMatch = MatchPoints(featuresFixed, featuresMove)
    pointsFixed = np.float32([keypointsFixed[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
    pointsMove = np.float32([keypointsMove[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

    # affine matrix
    H = ComputeH(pointsMove, pointsFixed)

    return H


def warpImage(im, H, outSize):
    imwarped = cv2.warpPerspective(im, H, outSize, flags=cv2.INTER_LINEAR)

    return imwarped
