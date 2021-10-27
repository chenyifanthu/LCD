import cv2
import numpy as np
from models.superpoint import SuperPointFrontend

print('==> Loading pre-trained network.')
# This class runs the SuperPoint network and processes its outputs.
fe = SuperPointFrontend(weights_path="pretrained/superpoint_v1.pth",
                        nms_dist=4, conf_thresh=0.015,
                        nn_thresh=0.7, cuda=True)
print('==> Successfully loaded pre-trained network.')

img = cv2.imread("samples/test.jpg", 0)
img = img.astype(np.float32)
pts, desc, heatmap = fe.run(img)
print(pts)
print(pts.shape)