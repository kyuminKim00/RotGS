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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, set_rasterizer
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.residual_predictor import ResidualPredictor
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, fixed_camera, residual_predictor, wo_tiny=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    raster_setting_flag = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if raster_setting_flag == 0:
            fixed_rasterizer = set_rasterizer(view, gaussians, pipeline, background, scaling_modifier=1.0)
            raster_setting_flag = 1

        if fixed_camera:
            coarse_angle = view.rotation_angle.clone().detach().unsqueeze(0).to('cuda')
            if wo_tiny is False:
                time_tensor = torch.tensor(view.time, dtype=torch.float).to('cuda')
                residual = residual_predictor(time_tensor, view.cam_idx)
                angle = coarse_angle + residual
            else:
                angle = coarse_angle

            axis = gaussians.get_axis(view.cam_idx)
            center = gaussians.get_center(view.cam_idx)
            rendering = render(fixed_rasterizer, gaussians, axis=axis, center=center, angle=angle)["render"]
        else:
            rendering = render(fixed_rasterizer, gaussians, axis=None, center=None, angle=None)["render"]

        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        if args.multi_camera:
            camera_dir = [d for d in os.listdir(args.source_path) if os.path.isdir(os.path.join(args.source_path, d))]
            number_of_cameras = len(camera_dir)
        else:
            number_of_cameras = 1
        gaussians = GaussianModel(dataset.sh_degree, fixed_camera=args.fixed_camera, multi_camera=args.multi_camera, number_of_cameras=number_of_cameras)
        scene = Scene(dataset, gaussians, args.fixed_camera, load_iteration=iteration, shuffle=False, random_init=True, multi_camera=args.multi_camera, eval_mode=True)
        residual_predictor = ResidualPredictor(number_of_cameras)
        residual_predictor.load_weights(dataset.model_path)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args.fixed_camera, residual_predictor, args.wo_tiny)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args.fixed_camera, residual_predictor, args.wo_tiny)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fixed_camera", action="store_true", default=True, help="Fix camera position")
    parser.add_argument("--moving_camera", action="store_true", default=False, help="Moving camera position")
    parser.add_argument("--wo_tiny", action="store_true", help="without tiny angle")
    parser.add_argument("--multi_camera", action="store_true", default=False, help="multi-camera system")
    parser.add_argument('--name', type=str, default='random', help='Output folder name')

    args = get_combined_args(parser)
    print(vars(args))
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.moving_camera == True:
        args.fixed_camera = False
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)