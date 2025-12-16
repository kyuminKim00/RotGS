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
from utils.general_utils import safe_state, plot_residual, plot_axis, save_comparison_image, plot_point_cloud
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):

    if args.multi_camera: # train multi monocular camera system
        camera_dir = [d for d in os.listdir(args.source_path) if os.path.isdir(os.path.join(args.source_path, d))]
        number_of_cameras = len(camera_dir)
        dataset.sh_degree = 1  # multi camera system -> level 1 sh only
    else:
        number_of_cameras = 1

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, output_name=args.name)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, args.fixed_camera, args.wo_axis, args.multi_camera, number_of_cameras)
    scene = Scene(dataset, gaussians, args.fixed_camera, args.random, args.multi_camera)
    gaussians.training_setup(opt)
    residual_predictor = ResidualPredictor(number_of_cameras)
    residual_predictor.train_setting(opt)

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
    chunk_size = len(viewpoint_stack) // number_of_cameras 
    viewpoint_stack = [viewpoint_stack[i*chunk_size : (i+1)*chunk_size] for i in range(number_of_cameras)]

    ema_loss_for_log = 0.0
    ema_rgbloss_for_log = 0.0
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    raster_setting_flag = 0

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        cam_idx = ((iteration-1) // args.camera_interval) % number_of_cameras
        if not viewpoint_stack[cam_idx]:
            new_cameras = scene.getTrainCameras().copy()
            chunk_size = len(new_cameras) // number_of_cameras
            viewpoint_stack = [new_cameras[i*chunk_size:(i+1)*chunk_size] for i in range(number_of_cameras)]
        if len(viewpoint_stack[cam_idx]) == 1:
            viewpoint_cam = viewpoint_stack[cam_idx].pop(0)
            new_cameras = scene.getTrainCameras().copy()
            chunk_size = len(new_cameras) // number_of_cameras
        else:
            viewpoint_cam = viewpoint_stack[cam_idx].pop(0)

        if raster_setting_flag == 0:
            fixed_rasterizer = set_rasterizer(viewpoint_cam, gaussians, pipe, bg, scaling_modifier=1.0)
            raster_setting_flag = 1

        gt_image = viewpoint_cam.original_image.cuda()

        # Render
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

        # rgb Loss
        rgb_loss = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)

         # total Loss, only rgb loss if multi camera system
        loss = (1.0 - opt.lambda_dssim) * rgb_loss + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        # for i, param_group in enumerate(gaussians.optimizer.param_groups):
        #     name = param_group.get("name", f"group{i}")
        #     for j, p in enumerate(param_group['params']):
        #         if p.grad is not None:
        #             print(f"{name} grad mean={p.grad.mean().item():.6f}")
                    
        iter_end.record()

        with torch.no_grad():
            axis = torch.stack([gaussians.get_axis(cam_idx) for cam_idx in range(number_of_cameras)])      # [num_cameras, 3]
            centers = torch.stack([gaussians.get_center(cam_idx) for cam_idx in range(number_of_cameras)]) # [num_cameras, 3]

            if iteration in [1, 3_000, 7_000, 10_000, 12_000, 15_000, 20_000, 25_000, 30_000]:
                plot_point_cloud(gaussians.get_xyz, iteration, filename=f"pointcloud/pcd_{iteration}.png", axis=axis, centers=centers, number_of_cameras=number_of_cameras)
                save_comparison_image(iteration, image, gt_image)
                plot_axis(axis, filename=f"graph/axis_{args.name}.png")
                residual_predictor.plot_residual()
            
            # if (iteration-1) % 300 == 0:
            #     plot_axis(axis, filename=f"graph/axis_{args.name}.png")
                # residual_predictor.plot_residual()
                # plot_flow(predicted_flow, flow_2d_gt, flow_alpha_mask)

            # if iteration < 7_000 :
            #     save_comparison_image(iteration, image, gt_image)
            #     # if (iteration-1) % 100 == 0:
                # plot_point_cloud(gaussians.get_xyz, iteration, filename=f"pointcloud/pcd_{iteration}.png", axis=axis)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_rgbloss_for_log = 0.4 * rgb_loss + 0.6 * ema_rgbloss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"loss": f"{ema_loss_for_log:.{4}f}", "rgb": f"{ema_rgbloss_for_log:.{4}f}"})
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 10_000, 12_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 10_000, 12_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--fixed_camera", action="store_true", default=True, help="Fix camera position")
    parser.add_argument("--random", action="store_true", default=True, help="Random pointcloud initialization")
    parser.add_argument("--sfm", action="store_true", default=False, help="SfM pointcloud initialization")
    parser.add_argument("--wo_tiny", action="store_true", default=False, help="without tiny angle")
    parser.add_argument("--wo_flow", action="store_true", default=True, help="without optical flow supervision")
    parser.add_argument("--wo_UDFS", action="store_true", default=False, help="without UDFS")
    parser.add_argument("--wo_axis", action="store_true", default=False, help="without UDFS")
    parser.add_argument("--multi_camera", action="store_true", default=True, help="multi-camera system")
    parser.add_argument('--name', type=str, default='exper', help='Output folder name')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    if args.sfm == True:
        args.random = False

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    start_time = datetime.now()

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    # All done
    print("\nTraining complete.")

    with open("duration_log.txt", "a") as f:
        f.write(f"{args.name}: {end_time-start_time}\n")