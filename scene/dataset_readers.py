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
from dataclasses import dataclass
import os
import sys
from copy import copy
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.general_utils import add_noise
from scene.gaussian_model import BasicPointCloud
import torch
import re

@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    time : float
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    rotation_angle : float
    cam_idx : int = 0
    flow_idx : int = 0


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    distance: float

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    return {"translate": translate, "radius": radius}

def readFixedCameras(cam_intrinsics, images_folder, test_cam_names_list, cam_idx=0):
    cam_infos = []
    images_paths = [i for i in os.listdir(images_folder) if i.endswith(('.jpg', '.png', 'jpeg'))]
    images_paths = sorted(images_paths)
    angle_array = torch.linspace(0, -2 * np.pi, steps=len(images_paths)) # single one full rotation (coarse angle)
    angle_array = add_noise(angle_array, noise_std=0.5, seed=1304) # to replicate residual noise, Add gaussian noise to coarse angle

    for idx, img in enumerate(images_paths):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}". format(idx+1, len(images_paths)))
        sys.stdout.flush()
        intr = cam_intrinsics[1] 
        height = intr.height
        width = intr.width
        match = re.search(r'(\d+)\.(jpg|png)$', img) # ex) image_name == 'web_cam012.jpg' -> time = 12
        time = int(match.group(1)) if match else None
        rotation_angle = angle_array[time]
        normalized_time = time / (len(images_paths) - 1)
        uid = intr.id
        
        R = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
            ])

        tvec = np.array([0, 0, 5])
        T = np.array(tvec)

        X_camera = R.T @ (-tvec)
        distance = np.linalg.norm(X_camera)  # distance from scene middle to camera
        
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
    
        image_path = os.path.join(images_folder, img)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=None, time=normalized_time,
                              image_path=image_path, image_name=img, depth_path="", width=width, height=height, is_test=image_path in test_cam_names_list,
                              rotation_angle=rotation_angle, cam_idx=cam_idx)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos, distance


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    N = len(vertices)
    def has_fields(fields):
        return all(name in vertices.data.dtype.names for name in fields)
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if has_fields(['red', 'green', 'blue']):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.zeros((N, 3), dtype=np.float32)
    if has_fields(['nx', 'ny', 'nz']):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros((N, 3), dtype=np.float32)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readFixedSceneInfo(path, images, eval, train_test_exp, random_init=True, llffhold=8, eval_mode=False): 
    # get intrinstic parameter
    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    if eval:
        llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            image_folder = os.path.join(path, "images")
            image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            image_files = sorted(image_files)
            test_cam_names_list = []
            for i in range(0, len(image_files), 8):  
                test_cam_names_list.append(image_files[i])
    else:
        test_cam_names_list = []
    
    # get images
    reading_dir = "images" if images == None else images
    cam_infos_unsorted, distance = readFixedCameras(
        cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = []
    test_cam_infos = []

    for i in cam_infos:
        if i.image_name in test_cam_names_list:
            test_cam_infos.append(i)
        else:
            train_cam_infos.append(i)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if random_init:
        # init from random points
        num_pts = 5_000
        print("random pointcloud initialization")
        radius = distance
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = (np.random.random((num_pts, 3)) * radius - (radius * 0.5)).astype(np.float32)
        shs = (np.random.random((num_pts, 3)) / 255.0).astype(np.float32)
        num_pts = xyz.shape[0]
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        ply_path = os.path.join(path, "sparse_colmap/0/points3D.ply")
        bin_path = os.path.join(path, "sparse_colmap/0/points3D.bin")
        txt_path = os.path.join(path, "sparse_colmap/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
            print("sfm pointcloud initialization")
        except:
           pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           is_nerf_synthetic=False,
                           distance=distance)
    return scene_info

def readMultiSceneInfo(path, images, eval, train_test_exp, random_init=True, llffhold=8, eval_mode=False): 
    camera_dir = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    camera_dir.sort()
    train_cam_infos = []
    test_cam_infos = []
    print("total Camera number: {}".format(len(camera_dir)))
    for idx, cam in enumerate(camera_dir):
        cam_idx = idx
        camera_path = os.path.join(path, cam)
        # get intrinstic parameter
        try:
            cameras_intrinsic_file = os.path.join(camera_path, "sparse/0", "cameras.bin")
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_intrinsic_file = os.path.join(camera_path, "sparse/0", "cameras.txt")
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        if eval:
            llffhold = 8
            if llffhold:
                image_folder = os.path.join(camera_path, "images")
                image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                image_files = sorted(image_files)
                test_cam_names_list = []
                for i in range(0, len(image_files), 8):  
                    test_cam_names_list.append(image_files[i])
        else:
            test_cam_names_list = []
    
        # get images
        reading_dir = "images" if images == None else images
        cam_infos_unsorted, distance = readFixedCameras(
            cam_intrinsics=cam_intrinsics, images_folder=os.path.join(camera_path, reading_dir), test_cam_names_list=test_cam_names_list, cam_idx=cam_idx)
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

        train_cam_infos_curr = []
        test_cam_infos_curr = []

        for i in cam_infos:
            if i.image_name in test_cam_names_list:
                test_cam_infos_curr.append(i)
            else:
                train_cam_infos_curr.append(i)
        
        train_cam_infos.extend(train_cam_infos_curr)
        test_cam_infos.extend(test_cam_infos_curr)

        nerf_normalization = getNerfppNorm(train_cam_infos)

    if random_init:
        # init from random points
        num_pts = 5_000
        print("random pointcloud initialization")
        radius = distance
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = (np.random.random((num_pts, 3)) * radius - (radius * 0.5)).astype(np.float32)
        shs = (np.random.random((num_pts, 3)) / 255.0).astype(np.float32)
        num_pts = xyz.shape[0]
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        ply_path = os.path.join(path, "sparse_colmap/0/points3D.ply")
        bin_path = os.path.join(path, "sparse_colmap/0/points3D.bin")
        txt_path = os.path.join(path, "sparse_colmap/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
            print("sfm pointcloud initialization")
        except:
           pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           is_nerf_synthetic=False,
                           distance=distance)
    return scene_info

sceneLoadTypeCallbacks = {
    "Fixed": readFixedSceneInfo,
    "Multi": readMultiSceneInfo
}