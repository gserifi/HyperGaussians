import argparse
import datetime
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from roma import euler_to_rotmat
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import Camera, GaussianModel, Scene_mica
from scene.dataset import CameraDataset
from src.deform_model import Deform_Model
from src.hypergaussians_deform_model import HyperGaussians_Deform_Model


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--idname", type=str, default="id1_25", help="id name")
    parser.add_argument("--logname", type=str, default="log", help="log name")
    parser.add_argument("--image_res", type=int, default=512, help="image resolution")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ours", action="store_true", default=False)
    parser.add_argument("--rotate", action="store_true", default=False)
    parser.add_argument("--uncertainty", action="store_true", default=False)
    parser.add_argument("--error", action="store_true", default=False)
    parser.add_argument("--cross", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    ## deform model
    if args.ours:
        DeformModel = HyperGaussians_Deform_Model(lpt, args.device).to(args.device)
    else:
        DeformModel = Deform_Model(lpt, args.device).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()

    ## dataloader
    data_dir = Path("dataset") / args.idname
    mica_datadir = Path("metrical-tracker/output") / args.idname
    logdir = Path("logs") / f"{args.idname}_{args.logname}"

    if args.cross is None:
        scene = Scene_mica(
            data_dir, mica_datadir, train_type=1, white_background=lpt.white_background, device=args.device
        )
    else:
        scene = Scene_mica(
            data_dir,
            mica_datadir,
            train_type=1,
            white_background=lpt.white_background,
            device=args.device,
            cross=args.cross,
        )
    dataloader = DataLoader(
        CameraDataset(scene.getTestCameras(), scene.bg_image, cross=args.cross is not None),
        batch_size=batch_size,
        shuffle=False,
        num_workers=lpt.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=lambda x: x,
    )
    dataloader_iter = iter(dataloader)

    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vid_save_path = logdir / "test.avi"
    if args.rotate:
        frames_save_path = logdir / "frames_rotate"
    else:
        frames_save_path = logdir / "frames"

    if args.uncertainty:
        frames_save_path = frames_save_path.parent / f"{frames_save_path.name}_uncertainty"

    if args.error:
        frames_save_path = frames_save_path.parent / f"{frames_save_path.name}_error"

    if args.cross is not None:
        frames_save_path = frames_save_path.parent / f"{frames_save_path.name}_cross"

    gt_frames_save_path = frames_save_path / "gt"
    render_frames_save_path = frames_save_path / "renders"
    mask_frames_save_path = frames_save_path / "mask"

    frames_save_path.mkdir(exist_ok=True)
    gt_frames_save_path.mkdir(exist_ok=True)
    render_frames_save_path.mkdir(exist_ok=True)
    mask_frames_save_path.mkdir(exist_ok=True)
    # out = cv2.VideoWriter(vid_save_path, fourcc, 25, (args.image_res * 2, args.image_res), True)

    viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict["shape"] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    cmap = torch.tensor(colormaps["viridis"].colors, device=args.device)

    angles = np.linspace(np.deg2rad(0), np.deg2rad(359), len(viewpoint) // 3)
    for iteration in tqdm(range(len(viewpoint)), desc="Rendering"):
        try:
            viewpoint_cam: Camera = next(dataloader_iter)[0]  # Batch size is 1
        except StopIteration:
            dataloader_iter = iter(dataloader)
            viewpoint_cam: Camera = next(dataloader_iter)[0]  # Batch size is 1

        viewpoint_cam.to(torch.device(args.device))
        # viewpoint_cam = viewpoint[iteration]
        frame_id = viewpoint_cam.uid

        # deform gaussians
        codedict["expr"] = viewpoint_cam.exp_param
        codedict["eyes_pose"] = viewpoint_cam.eyes_pose
        codedict["eyelids"] = viewpoint_cam.eyelids
        codedict["jaw_pose"] = viewpoint_cam.jaw_pose
        codedict["camera_center"] = viewpoint_cam.camera_center
        if lpt.conditional_appearance:
            verts_final, rot_delta, scale_coef, uncertainty, color, opacity = DeformModel.decode(codedict)
        else:
            verts_final, rot_delta, scale_coef, uncertainty = DeformModel.decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        if lpt.conditional_appearance:
            gaussians.update_clr_opc(color[0], opacity[0])

        if args.rotate:
            rotmat = euler_to_rotmat(
                convention="XYZ", angles=np.array([0.0, angles[iteration % angles.shape[0]], 0.0])
            ).numpy()
            viewpoint_cam.compute_transforms(rotmat @ viewpoint_cam.R, viewpoint_cam.T)
            viewpoint_cam.to(torch.device(args.device))

        # Render
        override_color = None
        if args.uncertainty:
            # high = torch.tensor([1.0, 1.0, 0.0], device=uncertainty.device)
            # low = torch.tensor([0.0, 1.0, 1.0], device=uncertainty.device)
            # uncertainty = uncertainty * high + (1 - uncertainty) * low
            uncertainty = torch.sigmoid(2.5 * uncertainty)  # 2.5 factor for better visualization
            uncertainty_long = (uncertainty * 255).long()
            override_color = cmap[uncertainty_long]
        elif lpt.conditional_appearance:
            override_color = gaussians.get_color
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            ppt,
            background,
            override_color=override_color,
        )

        image = render_pkg["render"]
        image = image.clamp(0, 1)

        gt_image = viewpoint_cam.original_image
        # save_image = np.zeros((args.image_res, args.image_res * 2, 3))

        if args.error:
            image = (gt_image - image).abs().mean(dim=0, keepdim=True) * 10
            image = torch.sigmoid(image).clamp(0, 1)
            image_long = (image.permute(1, 2, 0) * 255).long().squeeze(-1)
            image = cmap[image_long].permute(2, 0, 1) * viewpoint_cam.head_mask + (1 - viewpoint_cam.head_mask)
        gt_image_np = (gt_image * 255.0).permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image * 255.0).permute(1, 2, 0).detach().cpu().numpy()

        # save_image[:, : args.image_res, :] = gt_image_np
        # save_image[:, args.image_res :, :] = image_np
        # save_image = save_image.astype(np.uint8)
        # save_image = save_image[:, :, [2, 1, 0]]

        gt_image_pil = Image.fromarray(gt_image_np.astype(np.uint8))
        gt_image_pil.save(gt_frames_save_path / f"{iteration:05d}.png")

        image_pil = Image.fromarray(image_np.astype(np.uint8))
        image_pil.save(render_frames_save_path / f"{iteration:05d}.png")

        mask = viewpoint_cam.head_mask
        mask_np = (mask * 255.0).permute(1, 2, 0).detach().cpu().numpy()[..., 0]
        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L")
        mask_pil.save(mask_frames_save_path / f"{iteration:05d}.png")

    #     out.write(save_image)
    # out.release()
