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
import sys
from datetime import datetime
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import matplotlib.animation as animation
import time
from submodules.core_flow.utils_former.flow_viz import flow_to_image, flow_uv_to_colors
import imageio

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, min_steps=0
):
    """
    Continuous learning rate decay function with initial zero lr until min_step.
    After min_step, lr decays exponentially from lr_init to lr_final.
    """
   

    def helper(step):
        if step < min_steps:
            return 0.0
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip((step - min_steps) / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip((step - min_steps) / (max_steps - min_steps), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    

    return helper

def get_cosine_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, min_steps=0
):
    """
    Continuous learning rate decay function with initial zero lr until min_step.
    After min_step, lr decays exponentially from lr_init to lr_final.
    """
   
    def helper(step):
        if step >= max_steps:
            return 0.0
        if step < min_steps:
            return 0.0
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip((step - min_steps) / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip((step - min_steps) / (max_steps - min_steps), 0, 1)
        cosine_lr = lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(np.pi * t))
        return delay_rate * cosine_lr
    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L
def build_quaternion(R):
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    w = torch.sqrt(1 + trace) / 2
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)

    quaternions = torch.stack((w, x, y, z), dim=-1)
    return quaternions

def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=1)

def rotate_vector_by_quaternion(v, q):
        q_conj = q.clone()
        q_conj[:, 1:] = -q_conj[:, 1:]
        v_quat = torch.cat([torch.zeros((v.shape[0], 1), device=v.device), v], dim=1)
        v_rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
        return v_rotated[:, 1:]

def axis_angle2rotmat(axis, angle):
    ux, uy, uz = axis
    cos_t = torch.cos(angle)
    sin_t = torch.sin(angle)
    one_minus_cos = 1 - cos_t

    row0 = torch.stack([
        cos_t + ux**2 * one_minus_cos,
        ux*uy*one_minus_cos - uz*sin_t,
        ux*uz*one_minus_cos + uy*sin_t
    ], dim=0)
    row1 = torch.stack([
        uy*ux*one_minus_cos + uz*sin_t,
        cos_t + uy**2 * one_minus_cos,
        uy*uz*one_minus_cos - ux*sin_t
    ], dim=0)

    row2 = torch.stack([
        uz*ux*one_minus_cos - uy*sin_t,
        uz*uy*one_minus_cos + ux*sin_t,
        cos_t + uz**2 * one_minus_cos
    ], dim=0)

    rotation_matrix = torch.stack([row0, row1, row2], dim=0)
    rotation_matrix = rotation_matrix.squeeze(-1)
    return rotation_matrix

def get_pose_angle(axis, ref=None):
    if ref is None:
        ref = torch.tensor([0.0, 1, 0.0], device=axis.device, dtype=axis.dtype)

    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    ref  = ref  / (torch.norm(ref,  dim=-1, keepdim=True) + 1e-8)
    dot = torch.sum(ref * axis, dim=-1, keepdim=True)
    cross = torch.cross(ref, axis, dim=-1)
    sin = torch.norm(cross, dim=-1, keepdim=True)
    angle = torch.atan2(sin, dot)
    return angle

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def cartesian_to_spherical(ux, uy, uz):
    r = np.sqrt(ux**2 + uy**2 + uz**2)
    theta = np.arccos(uz / r)
    phi = np.arctan2(uy, ux)
    return theta, phi

def spherical_to_cartesian_GPU(theta, phi):  # input: CUDA tensor
    ux = torch.cos(phi) * torch.sin(theta)
    uy = torch.sin(phi) * torch.sin(theta)
    uz = torch.cos(theta)
    return ux, uy, uz

def spherical_to_cartesian(theta, phi): #input [0, 2phi)
    ux = np.cos(phi) * np.sin(theta)
    uy = np.sin(phi) * np.sin(theta)
    uz = np.cos(theta)
    ux = 0 if np.isnan(ux) else ux
    uy = 0 if np.isnan(uy) else uy
    uz = 0 if np.isnan(uz) else uz
    return ux, uy, uz

def inverse_activate_theta_phi(axis_direction):
    theta = axis_direction[0]
    phi = axis_direction[1]
    inverse_theta = torch.log(theta / torch.pi / (1 - theta / torch.pi))
    inverse_phi = torch.log(phi / (2 * torch.pi) / (1 - phi / (2 * torch.pi)))
    return torch.tensor([inverse_theta, inverse_phi]).float().cuda()

def centerize_pcd(pcd):
    center = np.mean(pcd, axis=0)
    pcd -= center
    pcd = torch.from_numpy(pcd)
    return pcd

def plot_point_cloud(pcd, iteration, filename="point_cloud.png", axis=None, centers=None, number_of_cameras=1, scale=4, fps=20, duration=10):
    """
    pcd: [N,3] point cloud tensor
    axes: [num_cameras, 3] tensor of camera axes
    centers: [num_cameras, 3] tensor of camera centers
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # point cloud
    pcd_cpu = pcd.detach().cpu().numpy() if isinstance(pcd, torch.Tensor) else np.array(pcd)
    pcd_cpu = centerize_pcd(pcd_cpu)
    x, y, z = pcd_cpu[:, 0], pcd_cpu[:, 1], pcd_cpu[:, 2]
    scatter = ax.scatter(x, y, z, s=1, marker='o', alpha=0.8)
    print("Number of points in pcd:", pcd_cpu.shape[0])

    # plot axes for each camera
    if axis is not None and centers is not None:
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] 
        num_colors = len(colors)
        axis_cpu = axis.detach().cpu().numpy() if isinstance(axis, torch.Tensor) else np.array(axis)
        centers_cpu = centers.detach().cpu().numpy() if isinstance(centers, torch.Tensor) else np.array(centers)
        for i in range(number_of_cameras):
            ux, uy, uz = axis_cpu[i]
            cx, cy, cz = centers_cpu[i]
            cx, cy, cz = 0, 0, 0
            color = colors[i % num_colors]
            ax.quiver(cx, cy, cz, ux, uy, uz, length=scale, color=color, arrow_length_ratio=0.1, linewidth=2)

    # axis labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Point Cloud Visualization")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    fig.text(0.5, 0.02, f"Iteration: {iteration}", ha='center', fontsize=12, color='black')

    # static save
    if filename.endswith(".png"):
        ax.view_init(elev=20, azim=45)
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        def update(frame):
            ax.view_init(elev=20, azim=frame)
            return scatter,
        frames = fps * duration
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
        if filename.endswith(".gif"):
            ani.save(filename, writer="pillow", fps=fps)
        elif filename.endswith(".mp4"):
            ani.save(filename, writer="ffmpeg", fps=fps)
        plt.close()

def save_image(rendered_image, output_folder, orbit=None):
        img = rendered_image.permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min()) * 255

        img = img.cpu().to(torch.uint8)
        img = Image.fromarray(img.numpy())

        file_count = len(os.listdir(output_folder))
        if orbit == None:
            file_name = f"image_{file_count +1}.png"
        else:
            file_name = f"image_{file_count + 1}_{orbit:.3f}.png"
        file_path = os.path.join(output_folder, file_name)
        img.save(file_path)

def plot_3d_gaussians(iteration, axis, angle, positions, rotations, scales, num_points=3):
    if isinstance(axis, torch.Tensor):
        axis = axis.detach().cpu().numpy()
    if isinstance(angle, torch.Tensor):
        angle = angle.detach().cpu().numpy()
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if isinstance(rotations, torch.Tensor):
        rotations = rotations.detach().cpu().numpy()
    if isinstance(scales, torch.Tensor):
        scales = scales.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    N = positions.shape[0]

    axis_center = [0, 0, 0]
    axis_direction = axis
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    theta = axis_direction[0]
    phi = axis_direction[1]
    ux = np.cos(phi) * np.sin(theta)
    uy = np.sin(phi) * np.sin(theta)
    uz = np.cos(theta)
    
    # make 3D Gaussian primitives
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    for i in range(N):
        sx, sy, sz = scales[i]
        ellipsoid = np.array([x.flatten() * sx, y.flatten() * sy, z.flatten() * sz])
        rot_matrix = R.from_quat(rotations[i]).as_matrix()
        rotated_ellipsoid = np.dot(rot_matrix, ellipsoid)
        ex = rotated_ellipsoid[0, :] + positions[i, 0]
        ey = rotated_ellipsoid[1, :] + positions[i, 1]
        ez = rotated_ellipsoid[2, :] + positions[i, 2]
        ex, ey, ez = ex.reshape(num_points, num_points), ey.reshape(num_points, num_points), ez.reshape(num_points, num_points)
        ax.plot_surface(ex, ey, ez, color=np.random.rand(3,), alpha=0.5)
    
    # Plot axis direction vector
    ax.quiver(axis_center[0], axis_center[1], axis_center[2],
              ux, uy, uz, color='r', length=10, linewidth=3, arrow_length_ratio=0.1, label='Axis Vector')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Top View of 3D Gaussians")
    ax.set_box_aspect([1, 1, 1])
    
    file_name = f"{angle:.6f}.png"

    file_path = os.path.join("gaussian_plot/", file_name)
    plt.savefig(file_path)
    plt.close(fig)

axis_history = {}  # {cam_idx: {'ux': [], 'uy': [], 'uz': []}}

def plot_axis(axis, filename):
    """
    axis: torch.Tensor or np.array of shape [num_cameras, 3]
    """
    if isinstance(axis, torch.Tensor):
        axis = axis.detach().cpu().numpy()
    
    num_cameras = axis.shape[0]

    for cam_idx in range(num_cameras):
        if cam_idx not in axis_history:
            axis_history[cam_idx] = {'ux': [], 'uy': [], 'uz': []}

    for cam_idx in range(num_cameras):
        ux, uy, uz = axis[cam_idx]
        axis_history[cam_idx]['ux'].append(ux)
        axis_history[cam_idx]['uy'].append(uy)
        axis_history[cam_idx]['uz'].append(uz)

    fig, axs = plt.subplots(num_cameras, 3, figsize=(18, 6*num_cameras))

    if num_cameras == 1:
        axs = axs[None, :]

    colors = ['c', 'm', 'y']
    for cam_idx in range(num_cameras):
        axs[cam_idx, 0].plot(axis_history[cam_idx]['ux'], marker='o', linestyle='-', color=colors[0])
        axs[cam_idx, 0].set_title(f'Camera {cam_idx} Ux over Iterations')
        axs[cam_idx, 0].set_xlabel('Iteration')
        axs[cam_idx, 0].set_ylabel('Ux')

        axs[cam_idx, 1].plot(axis_history[cam_idx]['uy'], marker='o', linestyle='-', color=colors[1])
        axs[cam_idx, 1].set_title(f'Camera {cam_idx} Uy over Iterations')
        axs[cam_idx, 1].set_xlabel('Iteration')
        axs[cam_idx, 1].set_ylabel('Uy')

        axs[cam_idx, 2].plot(axis_history[cam_idx]['uz'], marker='o', linestyle='-', color=colors[2])
        axs[cam_idx, 2].set_title(f'Camera {cam_idx} Uz over Iterations')
        axs[cam_idx, 2].set_xlabel('Iteration')
        axs[cam_idx, 2].set_ylabel('Uz')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

residual_values = []
def plot_residual(residual, output_folder='.'):
    if isinstance(residual, torch.Tensor):
        residual = residual.detach().cpu().numpy()
    residual_values.append(-residual)

    plt.figure(figsize=(10, 5))
    plt.plot(residual_values, marker='o', linestyle='-', color='r')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title('Residual over Iterations')
    plt.savefig(f"{output_folder}/graph/residual.png")
    plt.close()

def save_comparison_image(iteration, image, gt_image, save_path="comparison.png"):
    image_np = image.detach().permute(1, 2, 0).cpu().numpy()
    gt_image_np = gt_image.detach().permute(1, 2, 0).cpu().numpy()

    image_np = np.clip(image_np, 0, 1)
    gt_image_np = np.clip(gt_image_np, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Rendered Image")
    axes[0].axis("off")

    axes[1].imshow(gt_image_np)
    axes[1].set_title("Ground Truth Image")
    axes[1].axis("off")

    fig.text(0.5, 0.05, f"Iteration: {iteration}", ha='center', fontsize=12, color='black')

    plt.tight_layout()
    save_path = f"comparison/{iteration}.png"
    plt.savefig(save_path, dpi=80)
    plt.close()


def save_image(iteration, image, save_dir="image"):
    os.makedirs(save_dir, exist_ok=True)
    image_np = image.detach().permute(1, 2, 0).cpu().numpy()

    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    im = Image.fromarray(image_np)
    save_path = os.path.join(save_dir, f"{iteration}.png")
    im.save(save_path)

def add_noise(rotation_angle_array, noise_std=0.01, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if seed is None:
        seed = int(time.time() * 1000) % (2**32 - 1)
        torch.manual_seed(seed)

    T = rotation_angle_array.shape[0]
    
    final_noise = torch.randn(1) * noise_std
    accumulated_noise = torch.linspace(0, final_noise.item(), T)
    noisy_rotation_array = rotation_angle_array + accumulated_noise

    # x = list(range(T))
    # y = accumulated_noise.detach().cpu().numpy() * (180 / torch.pi)
    # z = noisy_rotation_array.detach().cpu().numpy() * (180 / torch.pi)
    # gt = rotation_angle_array.detach().cpu().numpy() * (180 / torch.pi)

    # plt.figure()
    # plt.plot(x, y, 'r')
    # plt.title("Monotonically Increasing Noise (Rotation Angle)")
    # plt.xlabel("Time")
    # plt.ylabel("Noise (degree)")
    # plt.savefig("graph/angle_noise.png")
    # plt.close()

    # plt.figure(figsize=(8, 8)) 
    # plt.plot(x, z, 'r', label="corase angle")
    # plt.plot(x, gt, 'b', label="fine angle")
    # plt.xlabel("Timestep", fontsize=26)
    # plt.ylabel("Angle(degree)", fontsize=26)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.legend(fontsize=32)
    # plt.savefig("graph/fine_angle.png", dpi=300)
    # plt.close()
    print("Noise Added!!!")
    return noisy_rotation_array


def flow_direction_to_image(flow_uv, convert_to_bgr=False):
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(u**2 + v**2) + 1e-5
    u = u / rad
    v = v / rad

    return flow_uv_to_colors(u, v, convert_to_bgr)

def flow_magnitude_to_image(flow_uv, max_flow=None, convert_to_bgr=False):
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    # magnitude
    mag = np.sqrt(u**2 + v**2)
    if max_flow is None:
        max_flow = np.max(mag) + 1e-5

    # normalize 0~1
    norm_mag = np.clip(mag / max_flow, 0, 1)

    # HSV -> RGB
    hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = norm_mag         # hue = magnitude (0~1) → rainbow
    hsv[..., 1] = 1.0              # saturation
    hsv[..., 2] = 1.0              # value

    import matplotlib.colors as mcolors
    rgb = mcolors.hsv_to_rgb(hsv)

    if convert_to_bgr:
        rgb = rgb[..., ::-1]

    return (rgb * 255).astype(np.uint8)


def plot_flow(predicted_flow, flow_2d_gt, flow_alpha_mask):
    acculmulated_flow = predicted_flow.sum(3) #(2, H, W, K) -> (2, H, W)
    # plot flow vector
    flow_img_pred = flow_to_image(acculmulated_flow.permute(1,2,0).detach().cpu().numpy(), convert_to_bgr=False)
    flow_img_gt = flow_to_image(flow_2d_gt.permute(1,2,0).detach().cpu().numpy(), convert_to_bgr=False)
    mask_np = flow_alpha_mask.detach().cpu().numpy().astype(np.uint8)  # (H,W)
    mask_np = mask_np[..., None]  # (H,W,1)
    h, w, c = flow_img_pred.shape
    white_bg = np.ones((h, w, c), dtype=np.uint8) * 255
    flow_img_pred_masked = np.where(mask_np == 1, flow_img_pred, white_bg)
    flow_img_gt_masked   = np.where(mask_np == 1, flow_img_gt, white_bg)
    imageio.imwrite('graph/pred_flow.png', flow_img_pred_masked)
    imageio.imwrite('graph/gt_flow.png', flow_img_gt_masked)

    # plot flow vector direction
    flow_img_pred = flow_direction_to_image(acculmulated_flow.permute(1,2,0).detach().cpu().numpy())
    flow_img_gt   = flow_direction_to_image(flow_2d_gt.permute(1,2,0).detach().cpu().numpy())
    mask_np = flow_alpha_mask.detach().cpu().numpy().astype(np.uint8)  # (H,W)
    mask_np = mask_np[..., None]  # (H,W,1)
    h, w, c = flow_img_pred.shape
    white_bg = np.ones((h, w, c), dtype=np.uint8) * 255
    flow_img_pred_masked = np.where(mask_np == 1, flow_img_pred, white_bg)
    flow_img_gt_masked   = np.where(mask_np == 1, flow_img_gt, white_bg)
    imageio.imwrite('graph/pred_flow_dir.png', flow_img_pred_masked)
    imageio.imwrite('graph/gt_flow_dir.png', flow_img_gt_masked)

    # plot flow vector magnitude
    flow_img_pred = flow_magnitude_to_image(acculmulated_flow.permute(1,2,0).detach().cpu().numpy())
    flow_img_gt   = flow_magnitude_to_image(flow_2d_gt.permute(1,2,0).detach().cpu().numpy())
    mask_np = flow_alpha_mask.detach().cpu().numpy().astype(np.uint8)  # (H,W)
    mask_np = mask_np[..., None]  # (H,W,1)
    h, w, c = flow_img_pred.shape
    white_bg = np.ones((h, w, c), dtype=np.uint8) * 255
    flow_img_pred_masked = np.where(mask_np == 1, flow_img_pred, white_bg)
    flow_img_gt_masked   = np.where(mask_np == 1, flow_img_gt, white_bg)
    imageio.imwrite('graph/pred_flow_mag.png', flow_img_pred_masked)
    imageio.imwrite('graph/gt_flow_mag.png', flow_img_gt_masked)