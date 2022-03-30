import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import utils

plt.close('all')

#%% Recover Homographies
imageNames = ['1.jpg', '2.jpg', '4.jpg']
fixName = '3.jpg'
imFixRGB = cv2.imread(os.path.join('Images', fixName))
imFixGray = cv2.cvtColor(imFixRGB, cv2.COLOR_BGR2GRAY)

imagesRBG, imagesGray = [], []
for f in imageNames:
    im = cv2.imread(os.path.join('Images', f))
    imagesRBG.append(im)
    imagesGray.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

# compute the transformation matrix
Hs = []
for im in imagesGray:
    Hs.append(utils.AffineMatrix(imFixGray, im))

#%% Warp the Images
xshift = 1400
yshift = 700
Hshift = np.array([[1, 0, xshift], [0, 1, yshift], [0, 0, 1]], dtype=np.float32) # shift matrix for adapting the view
targetImageSize = (3500, 2000)

imagesGrayWarped, imagesRGBWarped, imagesMaskWarped = [], [], []
for H, imGray, imRGB, f in zip(Hs, imagesGray, imagesRBG, imageNames):
    # warp gray image
    imGrayWarped = utils.warpImage(imGray, Hshift @ H, targetImageSize)
    imagesGrayWarped.append(imGrayWarped)
    # warp mask
    mask = imGray * 0 + 1
    maskWarped = utils.warpImage(mask, Hshift @ H, targetImageSize)
    imagesMaskWarped.append(maskWarped)
    # warp rgb image
    imRGBWarped = np.zeros([targetImageSize[1], targetImageSize[0], 3])
    for i in range(3):
        imRGBWarped[:, :, i] = utils.warpImage(imRGB[..., i], Hshift @ H, targetImageSize)
    imagesRGBWarped.append(imRGBWarped)

    cv2.imwrite('./Results/Warped_' + f[0] + '.png', np.dstack((imRGBWarped, 255 * maskWarped)))

# warp Fixed image
imFixGrayWarped = utils.warpImage(imFixGray, Hshift, targetImageSize)
mask = imFixGray * 0 + 1
imFixMaskWarped = utils.warpImage(mask, Hshift, targetImageSize)
imFixRGBWarped = np.zeros([targetImageSize[1], targetImageSize[0], 3])
for i in range(3):
    imFixRGBWarped[:, :, i] = utils.warpImage(imFixRGB[..., i], Hshift, targetImageSize)
cv2.imwrite('./Results/Warped_' + fixName[0] + '.png', np.dstack((imFixRGBWarped, 255 * imFixMaskWarped)))

#%% Blend the images into a mosaic
# compute fusion weights. eg. if three images overlapes in a area, the weights of in each image should be 1/3
Gains = imFixMaskWarped.astype(np.float32)
for mask in imagesMaskWarped:
    Gains += mask.astype(np.float32)
plt.imshow(Gains)
Weights = Gains
Weights[Gains > 0] = 1 / Gains[Gains > 0]

# Fusion
imageMosaic = np.zeros([targetImageSize[1], targetImageSize[0], 3])
for i, (imRGBWarped, imMaskWarped) in enumerate(zip(imagesRGBWarped, imagesMaskWarped)):
    for c in range(3):
        imageMosaic[:, :, c] += imMaskWarped * Weights * imRGBWarped[:, :, c].astype(np.float32)
for c in range(3):
    imageMosaic[:, :, c] += imFixMaskWarped * Weights * imFixRGBWarped[:, :, c].astype(np.float32)
cv2.imwrite('./Results/Mosaic.png', np.dstack((imageMosaic, 255 * (Gains > 0))))

#%% Illustrasion
# Draw tranformed bbox on the Mosaic image
imageShow = imageMosaic *1.0
for (H, imGray) in zip(Hs, imagesGray):
    h, w = imGray.shape
    points = np.array([[0, 0], [0, h], [w, h], [w, 0], [0, 0]], dtype=np.float32)
    pointsTrans = np.hstack([points, np.ones([points.shape[0], 1])]) @ (Hshift @ H).T
    pointsTrans = np.array(pointsTrans[:,:2] / pointsTrans[:, 2:3], dtype=np.int)
    for i in range(pointsTrans.shape[0]-1):
        cv2.line(imageShow, pointsTrans[i, :], pointsTrans[i + 1, :], color=(255, 0, 0), thickness=5)

h, w = imFixGray.shape
points = np.array([[0, 0], [0, h], [w, h], [w, 0], [0, 0]], dtype=np.float32)
pointsTrans = np.hstack([points, np.ones([points.shape[0], 1])]) @ Hshift.T
pointsTrans = np.array(pointsTrans[:, :2] / pointsTrans[:, 2:3], dtype=np.int)
for i in range(pointsTrans.shape[0] - 1):
    cv2.line(imageShow, pointsTrans[i, :], pointsTrans[i + 1, :], color=(0, 0, 255), thickness=5)
cv2.imwrite('./Results/MosaicBoundry.png', np.dstack((imageShow, 255 * (Gains > 0))))
