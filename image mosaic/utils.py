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
    # imwarped = cv2.warpPerspective(im, H, outSize, flags=cv2.INTER_LINEAR)

    imwarped = MywarpImage(im, H, outSize)

    return imwarped


# Bells 1
def MywarpImage(im, H, outSize):
    if len(im.shape) < 3:
        im = im[..., None]
    h, w, c = im.shape
    Wtarget, Htarget = outSize
    imwarped = np.zeros([outSize[1], outSize[0], im.shape[-1]])

    jj, ii = np.meshgrid(np.arange(Wtarget), np.arange(Htarget))

    coords = np.hstack((jj.flatten()[:, None], ii.flatten()[:, None], np.ones([Htarget * Wtarget, 1])))
    Hinv = np.linalg.inv(H)
    coordsraw = coords @ Hinv.T
    coordsraw = coordsraw / coordsraw[:, 2:3]
    xraw, yraw = coordsraw[:, 0], coordsraw[:, 1]

    flag = (xraw >= 0) & (xraw < w - 1) & (yraw >= 0) & (yraw < h - 1)

    xraw, yraw = xraw[flag], yraw[flag]
    x, y = np.floor(xraw).astype(np.int), np.floor(yraw).astype(np.int)
    dx, dy = (xraw - x)[:, None], (yraw - y)[:, None]
    jj, ii = coords[flag, 0].astype(np.int), coords[flag, 1].astype(np.int)

    im00 = im * 1.0
    im01 = im[:, 1:, :]
    im10 = im[1:, :, :]
    im11 = im[1:, 1:, :]

    imwarped[ii, jj, :] = im00[y, x, :]*(1-dx)*(1-dy) + \
                          im01[y, x, :]*(dx)*(1-dy) + \
                          im10[y, x, :]*(1-dx)*(dy) + \
                          im11[y, x, :]*(dx)*(dy)

    imwarped = np.clip(imwarped, 0, 255).astype(np.uint8)

    return imwarped.squeeze()

# Bells 2
def calculateIntrinsics(width, height, fov=60):
    focal_len = width / (2 * np.tan(fov))

    return np.array([[focal_len, 0, width / 2], [0, focal_len, height / 2], [0, 0, 1]])


def cylindricalWarp(img, K):
    foc_len = (K[0][0] + K[1][1]) / 2
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1], 0:img.shape[0]]
    x, y = temp[0], temp[1]
    color = img[y, x]
    theta = (x - K[0][2]) / foc_len  # angle theta
    h = (y - K[1][2]) / foc_len  # height
    p = np.array([np.sin(theta), h, np.cos(theta)])
    p = p.T
    p = p.reshape(-1, 3)
    image_points = K.dot(p.T).T
    points = image_points[:, :-1] / image_points[:, [-1]]
    points = points.reshape(img.shape[0], img.shape[1], -1)
    cylinder = cv2.remap(img, (points[:, :, 0]).astype(np.float32), (points[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR)
    
    return cylinder
