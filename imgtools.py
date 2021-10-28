import cv2
import torch
import numpy as np
from models.patchnet import PatchNetAutoencoder

def cut_patch(image, pt, patchsize):
    h, w = image.shape[0], image.shape[1]
    x, y = int(pt[0]), int(pt[1])
    if patchsize % 2 == 0:
        left, right = x - patchsize//2, x + patchsize // 2
        up, down = y - patchsize//2, y + patchsize // 2
    else:
        left, right = x - patchsize//2, x + patchsize // 2 + 1
        up, down = y - patchsize//2, y + patchsize // 2 + 1
    if left < 0 or right > w or up < 0 or down > h:
        return None
    patch = image[up:down, left:right]
    return patch

def extract_image_patches(image, detector, patchsize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kps_detect = detector.detect(gray, None)
    kps = []
    patches = []
    for kp in kps_detect:
        patch = cut_patch(image, kp.pt, patchsize)
        if patch is not None:
            kps.append(kp)
            patches.append(patch)
    patches = np.stack(patches, axis=0)
    return kps, patches


def compute_lcd_descriptors(patches, model, batch_size, device):
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    descriptors = []
    with torch.no_grad():
        for i, x in enumerate(batches):
            x = x.to(device)
            z = model.encode(x)
            z = z.cpu().numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)

if __name__ == '__main__':
    device = "cuda"
    fname = "pretrained/model.pth"
    print("> Loading model from {}".format(fname))
    model = PatchNetAutoencoder(256)
    model.load_state_dict(torch.load(fname)["patchnet"])
    model.to(device)
    model.eval()
    
    # image = cv2.imread("samples/test.jpg")
    # image = cv2.resize(image, (0,0), fx=1, fy=1)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # kps, patches = extract_image_patches(image, 
    #                                      detector=cv2.SIFT_create(),
    #                                      patchsize=64)

    # desc = compute_lcd_descriptors(patches, model, 512, device)
    # print(desc.shape)
    
    
    img1 = cv2.imread("samples/match1.jpg")
    img2 = cv2.imread("samples/match2.jpg")
    
    detector = cv2.SIFT_create()
    kp1, patches1 = extract_image_patches(img1, detector, patchsize=64)
    kp2, patches2 = extract_image_patches(img2, detector, patchsize=64)
    des1 = compute_lcd_descriptors(patches1, model, 512, device)
    des2 = compute_lcd_descriptors(patches2, model, 512, device)
    
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # sift = cv2.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(gray1, None)
    # kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.FlannBasedMatcher()
    # Match descriptors.
    m = bf.knnMatch(des1, des2, k=2)
    m = sorted(m, key = lambda x:x[0].distance)
    ok = [m1 for (m1, m2) in m if m1.distance < 0.8 * m2.distance]
    
    
    med = cv2.drawMatches(img1, kp1, img2, kp2, ok, None)
    med = cv2.resize(med, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("image", med)
    cv2.waitKey(0)
