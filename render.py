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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

#import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import imageio
import numpy as np

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips

# utils
from utils.logger import Logger as Log
from utils.init import init
from utils.basic_utils import str2bool, int2bool

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args_dict):
    path_dict = args_dict["path_dict"]
    render_path = os.path.join(path_dict["output_dir"], name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(path_dict["output_dir"], name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    is_training = False

    l1_test = 0.0
    psnr_test = 0.0
    lpips_test = 0.0
    ssim_test = 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, args_dict=args_dict, is_training=is_training)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
        gt =  torch.clamp(gt, 0.0, 1.0)

        l1_test += l1_loss(rendered_image, gt).mean().double()
        psnr_test += psnr(rendered_image, gt).mean().double()
        lpips_test += lpips(rendered_image, gt, net_type='vgg').mean().double()
        ssim_test += ssim(rendered_image, gt)

    psmr_sum = psmr_sum / len(views)
    ssim_sum = ssim_sum / len(views)
    l1_sum = l1_sum / len(views)
    lpips_sum = lpips_sum / len(views)
    Log.info("About the {}: the psnr is {:.6f}, the ssim is {:.6f}, the l1 is {:.6f}, the lpips is {:.6f}".format(name, psmr_sum, ssim_sum, l1_sum, lpips_sum))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args_dict):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, args_dict=args_dict)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args_dict)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args_dict)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    opti = OptimizationParams(parser)
    parser.add_argument('--config', type=str, default=None, help="path of config file")
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--distributed', type=int2bool, default=False, help="disable distributed training")
    parser.add_argument('--deterministic', type=bool, default=False, help="")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", default=True)

    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_gt", action="store_true")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6057)

    parser.add_argument('--logfile_level', default='info', type=str, help='To set the log level to files.')
    parser.add_argument('--stdout_level', default='info', type=str, help='To set the level to print to screen.')
    # parser.add_argument('--log_file', default="log/3dgs.log", type=str, dest='logging:log_file', help='The path of log files.')
    parser.add_argument('--rewrite', type=str2bool, nargs='?', default=False, help='Whether to rewrite files.')
    parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True,
                        help='Whether to write logging into files.')
    parser.add_argument('--log_format', type=str, nargs='?', default="%(asctime)s %(levelname)-7s %(message)s"
                        , help='Whether to write logging into files.')

    args = parser.parse_args(sys.argv[1:])
    torch.cuda.set_device(torch.device("cuda:0"))
    init(args, is_inference=True)
    model.from_cfg(args)
    pipeline.from_cfg(args)
    opti.from_cfg(args)
    del opti

    #safe_state(args.quiet)

    outdoor_scenes=['bicycle', 'flowers', 'garden', 'stump', 'treehill']
    indoor_scenes=['room', 'counter', 'kitchen', 'bonsai']
    for scene in outdoor_scenes:
        if scene in args.source_path:
            args.images = "images_4"
            Log.info("Using images_4 for outdoor scenes")
    for scene in indoor_scenes:
        if scene in args.source_path:
            args.images = "images_2"
            Log.info("Using images_2 for indoor scenes")
    if 'playroom' in args.source_path:
        args.images = "images"
        Log.info("reset to images")

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.__dict__)