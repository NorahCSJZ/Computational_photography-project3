import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance
import utils

plt.close('all')

image = cv2.imread('./Images/cube.png')
H, W = image.shape[:2]

# Show Image
plt.figure(figsize=(12, 8))
plt.imshow(image[:, :, [2, 1, 0]])
plt.axis('off')
plt.title('Click four corners TL-BL-BR-TR (Rectangle)')

# get four points from the image in the order（topleft - bottomleft - bottomright - topright）
pos = np.array(plt.ginput(4))
np.save('pos.npy', pos)
posShow = np.vstack((pos, pos[0, :]))
plt.plot(posShow[:, 0], posShow[:, 1], 'g')
plt.axis('off')
plt.savefig('./Results/S1 rectangle.png')

# affine the rectangle to the area of [w, h]
h = int(np.linalg.norm(pos[0, :] - pos[1, :]))
w = int(np.linalg.norm(pos[0, :] - pos[-1, :]))
tpos = np.array([[0, 0], [0, h], [w, h], [w, 0]])
M1 = utils.ComputeH(pos.astype(np.float32), tpos.astype(np.float32))
imageT = utils.warpImage(image, M1, [int(W), int(H)])
plt.figure()
plt.imshow(imageT[:, :, ::-1])
plt.axis('off')
plt.savefig('./Results/S2 warped.png')

# shift to the center and redefine the output image size
corner_ori = np.array([[0, 0], [0, H], [W, H], [W, 0]])
corner_tr = np.hstack([corner_ori, np.ones([4, 1])]) @ M1.T
corner_tr = corner_tr / corner_tr[:, 2:3]

H_tr = int(np.max(corner_tr[:, 1]) - np.min(corner_tr[:, 1]))
W_tr = int(np.max(corner_tr[:, 0]) - np.min(corner_tr[:, 0]))
yshift = -np.min(corner_tr[:, 1])
xshift = -np.min(corner_tr[:, 0])
M2 = np.array([[1, 0, xshift], [0, 1, yshift], [0, 0, 1]]) @ M1

imageT = utils.warpImage(image, M2, [W_tr, H_tr])

plt.figure()
plt.imshow(imageT[:, :, [2, 1, 0]])
plt.grid()
plt.axis('off')
plt.savefig('./Results/S3 shifted.png')
