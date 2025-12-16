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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, fixed_camera=False, random_init=False, 
                 multi_camera=False, load_iteration=None, shuffle=True, eval_mode=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        print("Loaded_iter", self.loaded_iter)
        self.gaussians = gaussians
        self.fixed_camera = fixed_camera
        self.multi_camera = multi_camera
        self.point_cloud = None
        self.max_time = 0
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if self.fixed_camera :
            if self.multi_camera:
                print("Multiple cameras system !")
                scene_info = sceneLoadTypeCallbacks["Multi"](args.source_path, args.images, args.eval, args.train_test_exp, random_init, eval_mode=eval_mode)
            else:
                print("use Fixed Monocular CAMERA !")
                scene_info = sceneLoadTypeCallbacks["Fixed"](args.source_path, args.images, args.eval, args.train_test_exp, random_init, eval_mode=eval_mode)
            gaussians.distance = scene_info.distance
        else:
            print("use Moving Monocular CAMERA !")
            assert False, "Use Vanila 3D Gaussian Splatting!"

        self.max_time = int((len(scene_info.train_cameras) + len(scene_info.test_cameras)) / gaussians.number_of_cameras)
        gaussians.max_time = self.max_time
        print("Max_time:", self.max_time)
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)
        self.point_cloud = scene_info.point_cloud

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
