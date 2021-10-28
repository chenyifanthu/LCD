import cv2
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm


def extract_pointcloud_patches(pcd, voxel_size, radius, num_points):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    downsampled = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(downsampled.points)
    patches = []
    for i in tqdm(range(points.shape[0])):
        k, index, _ = kdtree.search_hybrid_vector_3d(points[i], radius, num_points)
        if k < num_points:
            index = np.random.choice(index, num_points, replace=True)
        xyz = np.asarray(pcd.points)[index]
        rgb = np.asarray(pcd.colors)[index]
        xyz = (xyz - points[i]) / radius  # normalize to local coordinates
        patch = np.concatenate([xyz, rgb], axis=1)
        patches.append(patch)
    patches = np.stack(patches, axis=0)
    return downsampled, patches


def cut_image_patch(image, pt, patchsize):
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
    for kp in tqdm(kps_detect):
        patch = cut_image_patch(image, kp.pt, patchsize)
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
        for x in tqdm(batches):
            x = x.to(device)
            z = model.encode(x)
            z = z.cpu().numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)

