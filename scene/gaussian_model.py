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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, get_cosine_lr_func, build_rotation, axis_angle2rotmat, build_quaternion
from utils.general_utils import cartesian_to_spherical, spherical_to_cartesian, inverse_activate_theta_phi, get_pose_angle, quaternion_multiply, rotate_vector_by_quaternion
# from utils.reloc_utils import compute_relocation_cuda
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from e3nn import o3

class GaussianModel:
    def __init__(self, sh_degree, optimizer_type="default", fixed_camera=False, wo_axis=False, multi_camera=False, number_of_cameras=1):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.distance = None
        self.fixed_camera = fixed_camera
        self.wo_axis = wo_axis
        self.multi_camera = multi_camera
        self.number_of_cameras = number_of_cameras
        axis_init = torch.tensor([0.0, 1.0, 0.0])          # shape (3,)
        center_point_init = torch.tensor([0.0, 0.0, 0.0])  # shape (3,)
        self._axis = axis_init[None, :].repeat(self.number_of_cameras, 1)                   # shape (num_cameras, 3)
        self._center_point = center_point_init[None, :].repeat(self.number_of_cameras, 1)  # shape (num_cameras, 3)
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.axis_activation = lambda x: torch.nn.functional.normalize(x, p=2, dim=0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_exposure(self):
        return self._exposure
   
    def get_axis(self, cam_idx):
        return self.axis_activation(self._axis[cam_idx])
    
    def get_center(self, cam_idx):
        return self._center_point[cam_idx]
   
    def rotate_shs(self, shs_feat, rotation_matrix):
        rotation_matrix = rotation_matrix.detach()
        shs_dc = shs_feat[:, 0:1, :]
        shs_rest = shs_feat[:, 1:, :]

        device = rotation_matrix.device
        dtype = rotation_matrix.dtype

        P = torch.tensor([[0, 0, 1],
                        [1, 0, 0],
                        [0, 1, 0]], device=device, dtype=dtype)

        P_inv = P.T
        permuted_rotation_matrix = (P_inv @ rotation_matrix @ P).to('cpu') # Explicitly move the matrix to CPU because the e3nn requires internal CPU operations
        rot_angles = o3.matrix_to_angles(permuted_rotation_matrix)
        D_1 = o3.wigner_D(1, rot_angles[0]
                          , -rot_angles[1]
                          , rot_angles[2]).to(device=device, dtype=dtype) # move back to GPU
        # D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2]).to(device=device, dtype=dtype) # Uncomment for sh degree > 3
        # D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2]).to(device=device, dtype=dtype)
          #rotation of the shs features
        # one_degree_shs = shs_rest[:, :3]
        # one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
        # one_degree_shs = einsum(
        #         D_1,
        #         one_degree_shs,
        #         "... i j, ... j -> ... i",
        #     )
        # one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
        # shs_rest[:, :3] = one_degree_shs

        shs_rest[:, :3] = torch.matmul(D_1, shs_rest[:, :3])
        # shs_rest[:, 3:8] = torch.matmul(D_2, shs_rest[:, 3:8]) # Uncomment for sh degree > 3
        # shs_rest[:, 8:15] = torch.matmul(D_3, shs_rest[:, 8:15])

        return torch.cat([shs_dc, shs_rest], dim=1)
    
    def rotate_gaussian(self, axis, center, angle):  # axis : [ux, uy, uz]
        position = self.get_xyz
        rotation = self.get_rotation
        features = self.get_features

        rotmat = axis_angle2rotmat(axis, angle)
        angle_half = angle / 2
        ux, uy, uz = axis
        q_global = torch.stack([
            torch.cos(angle_half),
            ux * torch.sin(angle_half),
            uy * torch.sin(angle_half),
            uz * torch.sin(angle_half)
        ], dim=0)
        q_global = q_global.repeat(1, position.shape[0]).T

        new_position = position - center
        new_position = rotate_vector_by_quaternion(new_position, q_global)
        new_position = new_position + center

        new_rotation = quaternion_multiply(q_global, rotation)
        new_rotation = torch.nn.functional.normalize(new_rotation, p=2, dim=1)

        new_features = self.rotate_shs(features, rotmat)
        return new_position, new_rotation, new_features
    
    def multi_rotate_gaussian(self, axis, center, angle):
        position = self.get_xyz
        rotation = self.get_rotation
        features = self.get_features

        # pose transformation
        pose_angle = get_pose_angle(axis) 
        pose_axis = torch.tensor([1.0, 0.0, 0.0], device=position.device)
        pose_angle_half = pose_angle / 2
        px, py, pz = pose_axis
        q_pose = torch.stack([
            torch.cos(pose_angle_half),
            px * torch.sin(pose_angle_half),
            py * torch.sin(pose_angle_half),
            pz * torch.sin(pose_angle_half)
        ], dim=0)
        q_pose = q_pose.repeat(1, position.shape[0]).T
        rotmat_pose = axis_angle2rotmat(pose_axis, pose_angle)

        # make multi view
        rotate_angle_half = angle / 2
        rotate_axis = torch.tensor([0.0, 1.0, 0.0])
        ax, ay, az = rotate_axis
        q_rot = torch.stack([
            torch.cos(rotate_angle_half),
            ax * torch.sin(rotate_angle_half),
            ay * torch.sin(rotate_angle_half),
            az * torch.sin(rotate_angle_half)
        ], dim=0)
        q_rot = q_rot.repeat(1, position.shape[0]).T
        rotmat_rot = axis_angle2rotmat(rotate_axis, angle)

        q_total = quaternion_multiply(q_pose, q_rot)
        q_total = torch.nn.functional.normalize(q_total, p=2, dim=1)
        rotmat_total = torch.matmul(rotmat_pose, rotmat_rot)

        new_position = rotate_vector_by_quaternion(position, q_total)
        new_rotation = quaternion_multiply(q_total, rotation)
        new_rotation = torch.nn.functional.normalize(new_rotation, p=2, dim=1)
        new_features = self.rotate_shs(features, rotmat_total)

        return new_position, new_rotation, new_features

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        center_point = torch.tensor(np.mean(pcd.points, axis=0)).cuda()
        if self.multi_camera: # Camera numbers increase from front view to top view
            if self.number_of_cameras == 7: # axis initialization for multi-camera system
                axis = torch.tensor([[0,0.8,0.2],[0,0.8,0.2],[0,0.8,0.2],[0,0.71,0.71],[0,0.2,0.8],[0,0.2,0.8],[0,0.2,0.8]]).float().cuda()
            elif self.number_of_cameras == 6:
                axis = torch.tensor([[0,0.8,0.2],[0,0.8,0.2],[0,0.71,0.71],[0,0.71,0.71][0,0.2,0.8],[0,0.2,0.8]]).float().cuda()
            elif self.number_of_cameras == 5:
                axis = torch.tensor([[0,0.8,0.2],[0,0.8,0.2],[0,0.7,0.7],[0,0.2,0.8],[0,0.2,0.8]]).float().cuda()
            elif self.number_of_cameras == 4:
                axis = torch.tensor([[0,0.8,0.2],[0,0.8,0.2],[0,0.2,0.8],[0,0.2,0.8]]).float().cuda()
            elif self.number_of_cameras == 3:
                axis = torch.tensor([[0,0.8,0.2],[0,0.7,0.7],[0,0.8,0.2]]).float().cuda()
            else:  
                axis = torch.tensor([0, 0.7, 0.7]).float().cuda()
                axis = axis[None, :].repeat(self.number_of_cameras, 1) # shape (num_cameras, 3)
        else:
            axis = torch.tensor([[0, 1.0, 0.0]]).float().cuda() # initial axis(camera up vector), (single_camera)
        
        print(f"initial axis: {axis}")
        print(f"initial center point: {center_point}")

        center_point = center_point[None, :].repeat(self.number_of_cameras, 1)  # shape (num_cameras, 3)

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        if self.wo_axis:
            self._center_point = nn.Parameter(center_point, requires_grad=False)
            self._axis = nn.Parameter(axis, requires_grad=False)
        else:
            self._center_point = nn.Parameter(center_point, requires_grad=True)
            self._axis = nn.Parameter(axis, requires_grad=True)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense #0.01
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._axis], 'lr': training_args.center_axis_lr_init, "name": "axis"}, 
            {'params': [self._center_point], 'lr': training_args.center_axis_lr_init, "name": "center_point"}, 
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        self.axis_scheduler_args = get_cosine_lr_func(training_args.center_axis_lr_init, training_args.center_axis_lr_final,
                                                        max_steps=training_args.center_axis_lr_max_steps)
  

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                
            if param_group['name'] == "axis":
                lr = self.axis_scheduler_args(iteration)
                param_group['lr'] = lr

            if param_group['name'] == "center_point":
                lr = self.axis_scheduler_args(iteration)
                param_group['lr'] = lr    

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        axis = self._axis.detach().cpu().numpy()
        center = self._center_point.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        axis_path = os.path.join(os.path.dirname(path), "axis.npy")
        np.save(axis_path, axis)
        center_path = os.path.join(os.path.dirname(path), "center.npy")
        np.save(center_path, center)

    def reset_scale_rotation(self):
        dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
        scales_new = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rotation_new = torch.zeros((self.get_xyz.shape[0], 4), device="cuda")
        rotation_new[:, 0] = 1
        optimizable_tensors_scaling = self.replace_tensor_to_optimizer(scales_new, "scaling")
        optimizable_tensors_rot = self.replace_tensor_to_optimizer(rotation_new, "rotation")
        self._scaling = optimizable_tensors_scaling["scaling"]
        self._rotation = optimizable_tensors_rot["rotation"]

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)

        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

        axis_path = os.path.join(os.path.dirname(path), "axis.npy")
        if os.path.exists(axis_path):
            axis = np.load(axis_path)
            self._axis = nn.Parameter(torch.tensor(axis, dtype=torch.float, device="cuda").requires_grad_(True))

        center_path = os.path.join(os.path.dirname(path), "center.npy")
        if os.path.exists(center_path):
            center_point = np.load(center_path)
            self._center_point = nn.Parameter(torch.tensor(center_point, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "axis" or group["name"] == "center_point":
                continue 
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "axis" or group["name"] == "center_point":
                    continue 
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == "axis" or group["name"] == "center_point":
                continue 
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent) 
        
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if self.fixed_camera:
            extent = self.distance

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
        self.prune_points(prune_mask)
        self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    
    