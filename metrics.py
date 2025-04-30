#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the FlashAvatar_LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import lpips
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from utils.general_utils import normalize_for_percep
from utils.loss_utils import psnr, ssim


def readImages(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in tqdm(os.listdir(renders_dir), desc="Loading Images"):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = Image.open(mask_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0)[:, :1, :, :].cuda())
        image_names.append(fname)
    return renders, gts, masks, image_names


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    percep_module = lpips.LPIPS(net="vgg").to("cuda")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            frames_dir = Path(scene_dir) / "frames"
            if not frames_dir.exists():
                frames_dir = Path(scene_dir)

            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            gt_dir = frames_dir / "gt"
            renders_dir = frames_dir / "renders"
            mask_dir = frames_dir / "mask"
            renders, gts, masks, image_names = readImages(renders_dir, gt_dir, mask_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx] * masks[idx], gts[idx] * masks[idx]))
                psnrs.append(psnr(renders[idx] * masks[idx], gts[idx] * masks[idx]))

                image_percep = normalize_for_percep(renders[idx] * masks[idx])
                gt_image_percep = normalize_for_percep(gts[idx] * masks[idx])

                # They report numbers with the scaling factor, no idea why
                lpipss.append(torch.mean(percep_module.forward(image_percep, gt_image_percep)).item() * 0.05)

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir].update(
                {
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                }
            )
            per_view_dict[scene_dir].update(
                {
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                }
            )

            with open(scene_dir + "/results.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            raise e


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
