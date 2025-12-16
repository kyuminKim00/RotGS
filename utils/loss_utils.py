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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass
import math

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def compute_flow(proj_means_2D, proj_means_2D_next, gs_per_pixel, weight_per_gs_pixel, w, W, h, H):
    gs_per_pixel_down = F.interpolate(
                    gs_per_pixel.unsqueeze(0).float(),  # [1, 20, h, w]
                    size=(h, w),
                    mode='nearest'
                ).squeeze(0).long()
    weight_per_gs_pixel_down = F.interpolate(
                    weight_per_gs_pixel.unsqueeze(0),  # [1, 20, h, w]
                    size=(h, w),
                    mode='nearest'
                ).squeeze(0)
    
    predicted_flow = (proj_means_2D_next[gs_per_pixel_down] - proj_means_2D[gs_per_pixel_down]) * weight_per_gs_pixel_down.unsqueeze(-1)
    predicted_flow = predicted_flow.permute(3, 1, 2, 0)  # (K, h, w, 2) -> (2, h, w, K)
    predicted_flow[0]  *= w / W
    predicted_flow[1] *= h / H
    return predicted_flow

def L1_flow(predicted_flow, gt_flow, flow_alpha_mask):
    accumulated_flow = predicted_flow.sum(3) #(2, H, W, K) -> (2, H, W)
    flow_loss = torch.norm((gt_flow - accumulated_flow) * flow_alpha_mask, p=2, dim=0).mean() # (2, H, W)
    return flow_loss

def cosine_flow(predicted_flow, gt_flow, flow_alpha_mask, width, height):
    accumulated_flow = predicted_flow.sum(3) #(2, H, W, K) -> (2, H, W)
    accumulated_flow[0] /= width
    accumulated_flow[1] /= height

    gt_flow[0] /= width
    gt_flow[1] /= height
    accumulated_flow = F.normalize(accumulated_flow, dim=0, eps=1e-8)
    gt_flow = F.normalize(gt_flow, dim=0, eps=1e-8)

    accumulated_flow = accumulated_flow * flow_alpha_mask
    gt_flow = gt_flow * flow_alpha_mask
    
    cos_sim = (accumulated_flow * gt_flow).sum(dim=0)  # (H, W)
    return 1 - cos_sim.mean()


def flow_schedule(iteration, max_iter, gamma=5.0, warmup=3000):
    if iteration <= warmup:
        return 1.0
    progress = (iteration - warmup) / (max_iter - warmup)  # 0 ~ 1
    w = math.exp(-gamma * progress)
    return w
