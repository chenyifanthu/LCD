import os
import numpy as np
import open3d as o3d


from models.patchnet import PatchNetAutoencoder
from models.pointnet import PointNetAutoencoder
from tools import *


def load_model(model_path, device):
    print("> Loading model from {}".format(model_path))
    models = torch.load(model_path)

    img_model = PatchNetAutoencoder(256)
    img_model.load_state_dict(models["patchnet"])
    img_model.to(device)
    img_model.eval()

    pc_model = PointNetAutoencoder(256, 6, 6)
    pc_model.load_state_dict(models["pointnet"])
    pc_model.to(device)
    pc_model.eval()
    print("> Sucessfully load models")

    return img_model, pc_model


device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "pretrained/model.pth"
img_model, pc_model = load_model(model_path, device)

pcd = o3d.io.read_point_cloud("data/leica.ply")
image = cv2.imread("data/image3.jpg")
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

pcd_kp, pcd_patches = extract_pointcloud_patches(
    pcd, voxel_size=0.2, radius=0.5, num_points=1024
)

pcd_desc = compute_lcd_descriptors(
    pcd_patches, pc_model, batch_size=1024, device=device
)

detector = cv2.SIFT_create()
img_kp, img_patches = extract_image_patches(image, detector, patchsize=64)
img_desc = compute_lcd_descriptors(img_patches, img_model, 1024, device)

pcd_kp_ = np.array(pcd_kp.points)
img_kp_ = np.array([list(kp.pt) for kp in img_kp])

np.savez("data.npz", img_kp=img_kp_, img_desc=img_desc, pcd_kp=pcd_kp_, pcd_desc=pcd_desc)