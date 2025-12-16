#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from sklearn.cluster import KMeans
from utils.general_utils import axis_angle2rotmat

class BasicPointCloud:
    def __init__(self, points, colors=None, normals=None):
        self.points = np.array(points)
        self.colors = np.array(colors) if colors is not None else None
        self.normals = np.array(normals) if normals is not None else None

    @property
    def centerize(self):
        distances = np.linalg.norm(self.points, axis=1)
        threshold = np.max(distances) * 1.0
        valid_indices = distances <= threshold
        self.points = self.points[valid_indices]
        if self.colors is not None:
            self.colors = self.colors[valid_indices]
        if self.normals is not None:
            self.normals = self.normals[valid_indices]
        center = np.mean(self.points, axis=0)
        self.points -= center
        print("pointcloud centerized")
    
    @property
    def flip_up_down(self):
        self.points[:, 1] *= -1
        if self.normals is not None:
            self.normals[:, 1] *= -1

    def downsample(self, method='voxel', param=0.01):
        if method == 'voxel':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            if self.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(self.colors)
            if self.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(self.normals)
            pcd = pcd.voxel_down_sample(voxel_size=param)
            self.points = np.asarray(pcd.points).astype(np.float32)
            if self.colors is not None:
                self.colors = np.asarray(pcd.colors).astype(np.float32)
            if self.normals is not None:
                self.normals = np.asarray(pcd.normals).astype(np.float32)

        elif method == 'random':
            n = self.points.shape[0]
            sample_size = int(n * param)
            indices = np.random.choice(n, sample_size, replace=False)
            self.points = self.points[indices].astype(np.float32)
            if self.colors is not None:
                self.colors = self.colors[indices].astype(np.float32)
            if self.normals is not None:
                self.normals = self.normals[indices].astype(np.float32)

        elif method == 'kmeans':
            num_clusters = int(param)
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.points)
            self.points = kmeans.cluster_centers_.astype(np.float32)
            self.colors = None
            self.normals = None

        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        print(f"Pointcloud downsampled using '{method}' method.")

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_torch(R: torch.Tensor, t: torch.Tensor, translate: torch.Tensor = None, scale: float = 1.0):
    device = R.device
    dtype = R.dtype

    if translate is None:
        translate = torch.zeros(3, device=device, dtype=dtype)

    Rt = torch.eye(4, device=device, dtype=dtype)
    Rt[:3, :3] = R.transpose(0, 1) 
    Rt[:3, 3] = t

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt.float()

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def print_grad_hook(name):
    def hook(grad):
        print(f"{name} grad norm: {grad.norm().item():.6f}")
    return hook

def get_time_variant_pose(viewpoint_camera, axis, angle, center=None):
    axis.register_hook(print_grad_hook("axis"))
    angle.register_hook(print_grad_hook("angle"))   
    R_w2c = torch.from_numpy(viewpoint_camera.R).float().cuda()  # W2C
    T_w2c = torch.from_numpy(viewpoint_camera.T).float().cuda()  # W2C

    R_c2w = R_w2c.transpose(0, 1)
    C = -R_c2w @ T_w2c  # camera center in world
    axis_world = R_c2w @ axis
    axis_world = axis_world / torch.norm(axis_world)
    
    rotmat = axis_angle2rotmat(axis_world, angle)  # 3x3

    if center is None:
        center = torch.zeros_like(C).cuda()

    C_rotated = center + rotmat @ (C - center)
    R_c2w_rotated = rotmat @ R_c2w

    R_w2c_rotated = R_c2w_rotated.transpose(0, 1)
    T_w2c_rotated = -R_w2c_rotated @ C_rotated

    world_view_transform = getWorld2View2_torch(R_w2c_rotated, T_w2c_rotated).transpose(0, 1).cuda()

    projection_matrix = viewpoint_camera.projection_matrix
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    camera_center = torch.inverse(world_view_transform)[3, :3]

    return world_view_transform, full_proj_transform, camera_center
