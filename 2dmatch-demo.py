import cv2
import torch
import numpy as np
from models.patchnet import PatchNetAutoencoder
from tools import *


if __name__ == '__main__':
    device = "cuda"
    fname = "pretrained/model.pth"
    print("> Loading model from {}".format(fname))
    model = PatchNetAutoencoder(256)
    model.load_state_dict(torch.load(fname)["patchnet"])
    model.to(device)
    model.eval()
    
    
    img1 = cv2.imread("samples/match1.jpg")
    img2 = cv2.imread("samples/match2.jpg")
    
    # LCD
    detector = cv2.SIFT_create()
    kp1, patches1 = extract_image_patches(img1, detector, patchsize=64)
    kp2, patches2 = extract_image_patches(img2, detector, patchsize=64)
    des1 = compute_lcd_descriptors(patches1, model, 512, device)
    des2 = compute_lcd_descriptors(patches2, model, 512, device)
    
    # # SIFT
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # sift = cv2.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(gray1, None)
    # kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.FlannBasedMatcher()
    m = bf.knnMatch(des1, des2, k=2)
    m = sorted(m, key = lambda x:x[0].distance)
    ok = [m1 for (m1, m2) in m if m1.distance < 0.85 * m2.distance]
    
    
    med = cv2.drawMatches(img1, kp1, img2, kp2, ok, None)
    # med = cv2.resize(med, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite("samples/feature-match-sift.png", med)
    # cv2.imshow("image", med)
    # cv2.waitKey(0)
