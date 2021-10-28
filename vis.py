import cv2
import numpy as np
import open3d as o3d
from math import sin, cos, pi

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
    
pcd = o3d.io.read_point_cloud("/Users/chenyifan/Projects/PointCloudTools/data/20210808/leica.ply")
pcd_down = pcd.voxel_down_sample(0.05)
# # print(pcd)
img = cv2.imread("/Users/chenyifan/Projects/PointCloudTools/data/20210808/image3.jpeg")
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
h, w = img.shape[0], img.shape[1]

scale = 0.01
theta = - 0.8 * pi
img_pcd = image_to_pointcloud(img, scale, theta)
img_down = img_pcd.random_down_sample(0.2)

data = np.load("data.npz")
img_kp, img_desc = data["img_kp"], data["img_desc"]
pcd_kp, pcd_desc = data["pcd_kp"], data["pcd_desc"]

print(img_desc.shape, pcd_desc.shape)

bf = cv2.FlannBasedMatcher()
# Match descriptors.
matches = bf.knnMatch(img_desc, pcd_desc, k=2)
good = [m for (m, n) in matches if m.distance < 0.92 * n.distance]
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

o3d.visualization.draw_geometries([img_down, pcd_down, line_pcd])
