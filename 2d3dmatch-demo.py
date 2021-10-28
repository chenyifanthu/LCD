import os
import numpy as np
import open3d as o3d
from math import sin, cos, pi

from models.patchnet import PatchNetAutoencoder
from models.pointnet import PointNetAutoencoder
from tools import *


def load_model(model_path, device):
    print("> Loading model from {}".format(model_path))
    if device == "cpu":
        models = torch.load(model_path, map_location=torch.device('cpu'))
    else:
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


def image_to_pointcloud(image, scale, theta):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = image_rgb.reshape(-1, 3) / 255
    h, w = image.shape[0], image.shape[1]
    # print(h, w)
    x_, z_ = np.meshgrid(range(w), range(h))
    x = (x_.flatten() - w/2) * scale * cos(theta)
    y = (x_.flatten() - w/2) * scale * sin(theta)
    z = (h - 1 - z_.flatten()) * scale
    xyz = np.vstack([x, y, z]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    # print(pcd)
    return pcd



device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "pretrained/model.pth"
img_model, pc_model = load_model(model_path, device)

pcd = o3d.io.read_point_cloud("data/leica.ply")
image = cv2.imread("data/image3.jpeg")
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
h, w = image.shape[0], image.shape[1]

pcd_kp, pcd_patches = extract_pointcloud_patches(
    pcd, voxel_size=0.2, radius=0.15, num_points=1024
)

pcd_desc = compute_lcd_descriptors(
    pcd_patches, pc_model, batch_size=1024, device=device
)

detector = cv2.SIFT_create()
img_kp, img_patches = extract_image_patches(image, detector, patchsize=64)
img_desc = compute_lcd_descriptors(img_patches, img_model, 1024, device)

pcd_kp_ = np.array(pcd_kp.points)
img_kp_ = np.array([list(kp.pt) for kp in img_kp])

scale = 0.01
theta = - 0.8 * pi
img_pcd = image_to_pointcloud(image, scale, theta)
img_down = img_pcd.random_down_sample(0.2)

bf = cv2.FlannBasedMatcher()
# Match descriptors.
matches = bf.knnMatch(img_desc, pcd_desc, k=2)
good = [m for (m, n) in matches if m.distance < 0.9 * n.distance]
print(len(good))
points = []
lines = []
for match in good:
    train_idx = match.trainIdx
    query_idx = match.queryIdx
    x_, y_ = img_kp[query_idx]
    x = (x_-w/2)*scale*cos(theta) 
    y = (x_-w/2)*scale*sin(theta)
    z = (h-1-y_)*scale
    points.append([x,y,z])
    points.append(list(pcd_kp[train_idx]))
    lines.append([len(points)-2, len(points)-1])
    
# for i in range(len(points)):
#     if not i % 2:
#         print(points[i], asin(points[i][1]/points[i][0]))

colors = [[0, 0, 1] for i in range(len(lines))]
line_pcd = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([img_down, pcd_kp, line_pcd])

# np.savez("data.npz", img_kp=img_kp_, img_desc=img_desc, pcd_kp=pcd_kp_, pcd_desc=pcd_desc)