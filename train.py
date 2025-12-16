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
import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
from scene.residual_predictor import ResidualPredictor
import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, set_rasterizer
import sys
from scene import Scene, GaussianModel 
from utils.general_utils import safe_state, plot_residual, plot_axis, save_comparison_image, plot_point_cloud, plot_flow
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.loss_utils import compute_flow, L1_flow, cosine_flow, flow_schedule
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from submodules.gmflow.config import get_cfg as get_gmflow_cfg
from submodules.gmflow.gmflow import build_gmflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):

    number_of_cameras = 1 # single monocular camera system
    cam_idx = 0
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, output_name=args.name)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, args.fixed_camera, args.wo_axis, args.multi_camera, number_of_cameras)
    scene = Scene(dataset, gaussians, args.fixed_camera, args.random, args.multi_camera)
    gaussians.training_setup(opt)
    residual_predictor = ResidualPredictor(number_of_cameras)
    residual_predictor.train_setting(opt)

    if not args.wo_flow:
        print("Use optical flow supervision")
        cfg = get_gmflow_cfg()  
        flownet = torch.nn.DataParallel(build_gmflow(cfg)) 
        flownet = flownet.module
        gm_checkpoint = torch.load(cfg.model, map_location = 'cuda')
        weights = gm_checkpoint['model'] if 'model' in gm_checkpoint else gm_checkpoint
        flownet.load_state_dict(weights)
        flownet = flownet.cuda()
        flownet.eval()
        flow_2d_gt_list = []
    else:
        print("without optical flow supervision")

    if not args.wo_tiny:
        print("Use tiny angle to predict residual")
    else:
        print("without tiny angle to predict residual")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    ema_rgbloss_for_log = 0.0
    ema_flowloss_for_log = 0.0
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    flow_idx = 0
    raster_setting_flag = 0

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        elif len(viewpoint_stack) <= 1:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(0)
        viewpoint_cam_next = viewpoint_stack[0]

        if raster_setting_flag == 0:
            fixed_rasterizer = set_rasterizer(viewpoint_cam, gaussians, pipe, bg, scaling_modifier=1.0)
            raster_setting_flag = 1

        gt_image = viewpoint_cam.original_image.cuda()
        next_gt_image = viewpoint_cam_next.original_image.cuda()
        H, W = gt_image.shape[-2:]
        h, w = H // args.flow_downsampling, W // args.flow_downsampling # resized width, resized height(for flow)

        # Render
        if args.wo_flow:
            coarse_angle = viewpoint_cam.rotation_angle.unsqueeze(0).cuda()
            if not args.wo_tiny:
                time_tensor = torch.tensor(viewpoint_cam.time, dtype=torch.float32).to(device)
                residual = residual_predictor(time_tensor, cam_idx)
                angle = coarse_angle + residual
            else:
                angle = coarse_angle
            axis = gaussians.get_axis(cam_idx)
            center = gaussians.get_center(cam_idx)
            render_pkg = render(fixed_rasterizer, gaussians, axis=axis, center=center, angle=angle)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # use optical flow   
        else: 
            coarse_angle1 = viewpoint_cam.rotation_angle.unsqueeze(0).cuda()
            coarse_angle2 = viewpoint_cam_next.rotation_angle.unsqueeze(0).cuda()

            if not args.wo_tiny:
                time_tensor1 = torch.tensor(viewpoint_cam.time, dtype=torch.float32).to(device)
                time_tensor2 = torch.tensor(viewpoint_cam_next.time, dtype=torch.float32).to(device)
                residual1 = residual_predictor(time_tensor1, cam_idx)
                residual2 = residual_predictor(time_tensor2, cam_idx).detach()
                angle1 = coarse_angle1 + residual1
                angle2 = coarse_angle2 + residual2
            else:
                angle1 = coarse_angle1
                angle2 = coarse_angle2

            axis = gaussians.get_axis(cam_idx)
            center = gaussians.get_center(cam_idx)
            render_pkg = render(fixed_rasterizer, gaussians, axis=axis, center=center, angle=angle1)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            proj_means_2D, gs_per_pixel, weight_per_gs_pixel = render_pkg["proj_means_2D"], render_pkg["gs_per_pixel"].long(), render_pkg["weight_per_gs_pixel"].detach()

            render_pkg_next = render(fixed_rasterizer, gaussians, axis=axis, center=center, angle=angle2)
            proj_means_2D_next = render_pkg_next["proj_means_2D"].detach()
            
            predicted_flow = compute_flow(proj_means_2D, proj_means_2D_next, gs_per_pixel, weight_per_gs_pixel, w, W, h, H)

        if not args.wo_flow:
            if iteration < len(scene.getTrainCameras()):
                with torch.no_grad():
                    gt_image_flow = gt_image
                    next_gt_image_flow = next_gt_image
                    threshold = 0.03 
                    diff = torch.mean(torch.abs(gt_image - next_gt_image), dim=0)
                    mask = (diff > threshold).float()

                    flow_2d_gt = flownet(gt_image_flow[None]*255, next_gt_image_flow[None]*255) # return flow_predictions, feat_s, feat_t (540x540)
                    H_flow, W_flow = flow_2d_gt[0].shape[-2:]
                    
                    flow_alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    flow_alpha_mask = flow_alpha_mask * mask

                    if W_flow == w and H_flow == h:
                        flow_2d_gt = flow_2d_gt[0].squeeze() # (2, h, w)
                    else: 
                        flow_2d_gt = torch.nn.functional.interpolate(flow_2d_gt[0], size=(h, w), mode="bilinear").squeeze()
                        flow_2d_gt[0] *= w / W_flow
                        flow_2d_gt[1] *= h / H_flow # (135x135)
                        flow_alpha_mask = flow_alpha_mask.unsqueeze(0).unsqueeze(0).float()
                        flow_alpha_mask = F.interpolate(flow_alpha_mask, size=(h, w), mode="bilinear")
                        flow_alpha_mask = flow_alpha_mask.squeeze(0).squeeze(0)

                    viewpoint_cam.alpha_mask = flow_alpha_mask
                    flow_2d_gt = flow_2d_gt * flow_alpha_mask
                    flow_2d_gt_list.append(flow_2d_gt)
                    viewpoint_cam.flow_idx = flow_idx
                    flow_idx += 1
            else:
                flow_2d_gt = flow_2d_gt_list[viewpoint_cam.flow_idx] 
                flow_alpha_mask = viewpoint_cam.alpha_mask
        # rgb Loss
        rgb_loss = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)

        # flow Loss
        if not args.wo_flow and iteration < args.center_axis_lr_max_steps:
            if args.wo_UDFS == False:
                weight_udfs = flow_schedule(iteration, max_iter=args.center_axis_lr_max_steps, gamma=args.flow_schedule_lambda, warmup=args.warmup_iteration)
                flow_dir_loss = cosine_flow(predicted_flow, flow_2d_gt.detach().clone(), flow_alpha_mask, w, h)
                flow_mag_dir_loss = L1_flow(predicted_flow, flow_2d_gt.detach().clone(), flow_alpha_mask)
                flow_loss = weight_udfs * flow_dir_loss + (1 - weight_udfs) * flow_mag_dir_loss
            else:
                flow_mag_dir_loss = L1_flow(predicted_flow, flow_2d_gt.detach().clone(), flow_alpha_mask)
                flow_loss = flow_mag_dir_loss

        else:
            flow_loss = 0

         # total Loss
        loss = (1.0 - opt.lambda_dssim) * rgb_loss + opt.lambda_dssim * (1.0 - ssim_value) + args.lambda_flow * flow_loss
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_rgbloss_for_log = 0.4 * rgb_loss + 0.6 * ema_rgbloss_for_log
            ema_flowloss_for_log = 0.4 * flow_loss + 0.6 * ema_flowloss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"loss": f"{ema_loss_for_log:.{4}f}", "rgb": f"{ema_rgbloss_for_log:.{4}f}", "flow": f"{ema_flowloss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, rgb_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, gaussians, scene, render, 
                            fixed_rasterizer, dataset.train_test_exp, residual_predictor, args.wo_tiny)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                residual_predictor.save_weights(dataset.model_path, iteration)
            
            # 3DGS ADC
            if iteration < opt.densify_until_iter: #15,000

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) 
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: # 500 / 300
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None # 3,000
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none = True)
                residual_predictor.optimizer.step()
                residual_predictor.update_learning_rate(iteration)
                residual_predictor.optimizer.zero_grad(set_to_none = True)
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args, output_name='random'):  
    if output_name == 'random':  
        if not args.model_path:
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
    else:
        args.model_path = os.path.join("./output/", output_name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, gaussians, scene : Scene, renderFunc, rasterizer, train_test_exp, residual_predictor, wo_tiny):
    # Report test and samples of training set
    if iteration in testing_iterations: #[7000, 30000]
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    coarse_angle = (viewpoint.rotation_angle.unsqueeze(0).cuda())
                    if wo_tiny is False:
                        time_tensor = torch.tensor(viewpoint.time, dtype=torch.float32).to(device)
                        residual = residual_predictor(time_tensor, viewpoint.cam_idx)
                        angle = coarse_angle + residual
                    else:
                        angle = coarse_angle
                    axis = gaussians.get_axis(viewpoint.cam_idx)
                    center = gaussians.get_center(viewpoint.cam_idx)

                    image = torch.clamp(renderFunc(rasterizer, scene.gaussians, axis=axis, center=center, angle=angle)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
       
        torch.cuda.empty_cache()

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--fixed_camera", action="store_true", default=True, help="Fix camera position")
    parser.add_argument("--random", action="store_true", default=True, help="Random pointcloud initialization")
    parser.add_argument("--sfm", action="store_true", default=False, help="SfM pointcloud initialization")
    parser.add_argument("--wo_tiny", action="store_true", default=False, help="without tiny angle")
    parser.add_argument("--wo_flow", action="store_true", default=False, help="without optical flow supervision")
    parser.add_argument("--wo_UDFS", action="store_true", default=False, help="without UDFS")
    parser.add_argument("--wo_axis", action="store_true", default=False, help="without UDFS")
    parser.add_argument("--multi_camera", action="store_true", default=False, help="multi-camera system")
    parser.add_argument('--name', type=str, default='exper', help='Output folder name')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    if args.sfm == True:
        args.random = False

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")

