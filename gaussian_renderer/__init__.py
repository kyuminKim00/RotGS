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

from utils.graphics_utils import get_time_variant_pose
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

def set_rasterizer(viewpoint_camera, pc:GaussianModel, pipe, bg_color:torch.Tensor, scaling_modifier=1.0):
    """
    Background tensor (bg_color) must be on GPU!
    """
    viewmatrix=viewpoint_camera.world_view_transform 
    projmatrix=viewpoint_camera.full_proj_transform
    campos=viewpoint_camera.camera_center

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer

def render(rasterizer, pc:GaussianModel, axis=[0,1,0], center=[0,0,0], angle=0):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    cov3D_precomp = None
    scales = pc.get_scaling

    rotations = pc.get_rotation
    shs = pc.get_features
    colors_precomp = None

    if pc.fixed_camera:
        if pc.multi_camera:
            means3D, rotations, shs = pc.multi_rotate_gaussian(axis, center, angle)
        else:
            means3D, rotations, shs = pc.rotate_gaussian(axis, center, angle)

    rendered_image, radii, rendered_depth, rendered_alpha, proj_means_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu, weight_per_gaussian = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_depth": rendered_depth,
            "rendered_alpha": rendered_alpha,
             "proj_means_2D": proj_means_2D, 
             "conic_2D": conic_2D, 
             "conic_2D_inv": conic_2D_inv, 
             "gs_per_pixel": gs_per_pixel, 
             "weight_per_gs_pixel": weight_per_gs_pixel, 
             "x_mu": x_mu,
             "weight_per_gaussian": weight_per_gaussian}
    
    