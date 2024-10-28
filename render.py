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
import torch
from torch import nn
from utils.loss_utils import l1_loss, ssim, msssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, knn
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import make_grid
import numpy as np
import torchvision
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from os import makedirs
import imageio
import numpy as np
import glob
import collections
import math
import cv2
from typing import Optional
from scipy import signal
import lpips

def compute_psnr(img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
    """Compute PSNR between two images.

    Args:
        img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
    Returns:
        jnp.ndarray: PSNR in dB of shape ().
    """
    mse = (img0 - img1) ** 2
    return -10.0 / math.log(10)*torch.log(mse.mean())

def compute_ssim(
    # img0: jnp.ndarray,
    img0: torch.Tensor,
    # img1: jnp.ndarray,
    img1: torch.Tensor,
    # mask: Optional[jnp.ndarray] = None,
    mask: Optional[torch.Tensor] = None,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
# ) -> jnp.ndarray:
) -> torch.Tensor:
    """Computes SSIM between two images.

    This function was modeled after tf.image.ssim, and should produce
    comparable output.

    Image Inpainting for Irregular Holes Using Partial Convolutions.
        Liu et al., ECCV 2018.
        https://arxiv.org/abs/1804.07723

    Note that the mask operation is implemented as partial convolution. See
    Section 3.1.

    Args:
        img0 (jnp.ndarray): An image of size (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of size (H, W, 3) in float32.
        mask (Optional[jnp.ndarray]): An optional forground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.
        max_val (float): The dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
        filter_size (int): Size of the Gaussian blur kernel used to smooth the
            input images.
        filter_sigma (float): Standard deviation of the Gaussian blur kernel
            used to smooth the input images.
        k1 (float): One of the SSIM dampening parameters.
        k2 (float): One of the SSIM dampening parameters.

    Returns:
        jnp.ndarray: SSIM in range [0, 1] of shape ().
    """

    img0 = torch.as_tensor(img0).detach().cpu()
    img1 = torch.as_tensor(img1).detach().cpu()
    

    if mask is None:
        # mask = jnp.ones_like(img0[..., :1])
        mask = torch.ones_like(img0[..., :1])
    mask = mask[..., 0]  # type: ignore

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    # f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    f_i = ((torch.arange(filter_size).cpu() - hw + shift) / filter_sigma) ** 2
    # filt = jnp.exp(-0.5 * f_i)
    filt = torch.exp(-0.5 * f_i)
    # filt /= jnp.sum(filt)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # NOTICE Dusan: previous version used vectorization on Color channel, we need to avoid this
    def convolve2d(z, m, f):
        z_ = []
        for i in range(3):
            z_.append(torch.as_tensor(signal.convolve2d(z[...,i] * m, f, mode="valid")).cpu())
        z_ = torch.stack(z_, axis=-1)

        m_ = torch.as_tensor(signal.convolve2d(m, torch.ones_like(f), mode="valid")).cpu()

        return_where = []
        for i in range(3):
            return_where.append(torch.where(m_ != 0, z_[...,i] * torch.ones_like(f).sum() / m_, torch.tensor(0., device='cpu')))
        
        return_where = torch.stack(return_where, axis=-1)

        return return_where, (m_ != 0).type(z.dtype)

    filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
    filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])

    # Vmap the blurs to the tensor size, and then compose them.
    filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

    mu0 = filt_fn(img0, mask)[0]
    mu1 = filt_fn(img1, mask)[0]
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2, mask)[0] - mu00
    sigma11 = filt_fn(img1**2, mask)[0] - mu11
    sigma01 = filt_fn(img0 * img1, mask)[0] - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    # sigma00 = jnp.maximum(0.0, sigma00)
    sigma00 = torch.maximum(torch.tensor(0.0).cpu(), sigma00)
    # sigma11 = jnp.maximum(0.0, sigma11)
    sigma11 = torch.maximum(torch.tensor(0.0).cpu(), sigma11)
    # sigma01 = jnp.sign(sigma01) * jnp.minimum(
        # jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01)
    # )
    sigma01 = torch.sign(sigma01) * torch.minimum(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = ssim_map.mean()

    return ssim


def eval_imgs(pred, gt, loss_fn_vgg, scale_ssim=100., scale_lpips=100.):
    pred = torch.from_numpy(pred).float()/255. # H,W,3
    gt = torch.from_numpy(gt).float()/255. # H,W,3
    pred = pred.cuda()
    gt = gt.cuda()

    metric_psnr = compute_psnr(pred, gt).cpu()
    metric_ssim = compute_ssim(pred, gt).cpu() * scale_ssim
    metric_lpips = eval_lpips(pred, gt, loss_fn_vgg) * scale_lpips
    return dict(psnr=metric_psnr, ssim=metric_ssim, lpips=metric_lpips)

def eval_lpips(img0, img1, loss_fn_vgg):
    # normalize images from [0,1] range to [-1,1]
    img0 = img0 * 2.0 - 1.0
    img1 = img1 * 2.0 - 1.0
    img0 = img0.unsqueeze(0).permute(0, 3, 1, 2)
    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    return loss_fn_vgg(img0, img1).cpu()

@torch.no_grad()
def eval_all(src_dir, scale_ssim=100., scale_lpips=100.):
    results = collections.defaultdict(list)
    gt_dir = os.path.join(src_dir, 'gt')
    pred_dir = os.path.join(src_dir, 'renders')
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda().eval()

    gt_img_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')) + glob.glob(os.path.join(gt_dir, '*.jpg')))
    pred_img_paths = sorted(glob.glob(os.path.join(pred_dir, '*.png')) + glob.glob(os.path.join(pred_dir, '*.jpg')))
    assert len(gt_img_paths) == len(pred_img_paths), f'Number of images in gt and pred directories do not match: {len(gt_img_paths)} vs {len(pred_img_paths)}'

    for gt_img_path, img_path in tqdm(zip(gt_img_paths, pred_img_paths), total=len(gt_img_paths)):
        assert os.path.basename(gt_img_path) == os.path.basename(img_path), f'Image names do not match: {gt_img_path} vs {img_path}'
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_img_path)
        _eval = eval_imgs(img, gt, loss_fn_vgg, scale_ssim=scale_ssim, scale_lpips=scale_lpips)
        for key, val in _eval.items():
            results[key].append(val)
    for key, val in results.items():
        print(key, '=', torch.stack(val).mean().item())

    dst_results = os.path.join(src_dir, 'results.yaml')
    with open(dst_results, 'w') as f:
        f.write(f'ssim: {torch.stack(results["ssim"]).mean().item()}\n')
        f.write(f'psnr: {torch.stack(results["psnr"]).mean().item()}\n')
        f.write(f'lpips: {torch.stack(results["lpips"]).mean().item()}\n')
    print('Saved results to', dst_results)

@torch.no_grad()
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    results_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    renderings, gts = [], []
    for idx, view_img in enumerate(tqdm(views, desc="Rendering progress")):
        view = view_img[1].cuda()
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]

        gt = view.image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        renderings.append(to8b(rendering.cpu().numpy()))
        gts.append(to8b(gt.cpu().numpy()))
    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)
    gts = np.stack(gts, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(gts_path, 'video.mp4'), gts, fps=30, quality=8)
    # imageio.mimwrite(os.path.join(depth_path, 'video.mp4'), depths, fps=30, quality=8)
    print('Saved', os.path.join(render_path, 'video.mp4'))
    # evaluate test images
    # if name == 'test':
    #     eval_all(results_path)

def get_latest_ckpt(model_path):
    return os.path.join(model_path, "chkpnt_best.pth")
    ckpt_path_list = sorted(glob.glob(os.path.join(model_path, "chkpnt*0.pth")))
    max_iter = -1
    last_ckpt = None
    for ckpt_path in ckpt_path_list:
        basename = os.path.basename(ckpt_path).split('.')[0]
        basename = basename.split('chkpnt')[-1]
        iter = int(basename)
        if iter > max_iter:
            max_iter = iter
            last_ckpt = ckpt_path

    return last_ckpt

@torch.no_grad()
def render_sets(dataset: ModelParams, pipeline: PipelineParams, opt: OptimizationParams):
    ckpt_path = get_latest_ckpt(dataset.model_path)
    print(ckpt_path)
    ckpt = torch.load(ckpt_path)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipeline.eval_shfs_4d else 0)
    # gaussians.create_from_pth(ckpt_path, 1.0)
    gaussians.restore(ckpt[0],opt)
    scene = Scene(dataset, gaussians, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # render_set(dataset.model_path, "train", scene.loaded_iter,
    #             scene.train_cameras[1.0].copy(), gaussians, pipeline,
    #             background)

    render_set(dataset.model_path, "test", scene.loaded_iter,
                scene.test_cameras[1.0], gaussians, pipeline,
                background)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_pred", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])

    parser.add_argument("--config", type=str)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument('--num_pts', type=int, default=100_000)
    parser.add_argument('--num_pts_ratio', type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--exhaust_test", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
        
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
        
    if args.exhaust_test:
        args.test_iterations = args.test_iterations + [i for i in range(0,op.iterations,500)]
    
    setup_seed(args.seed)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    render_sets(lp.extract(args), pp.extract(args), op.extract(args))

    # All done
    print("\nTraining complete.")
# python train.py --config configs/resfields/views10_dancer.yaml