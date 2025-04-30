import argparse
import os
import random
import sys
import time
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene_mica
from scene.cameras import Camera
from scene.dataset import CameraDataset
from src.deform_model import Deform_Model
from src.hypergaussians_deform_model import HyperGaussians_Deform_Model
from utils.general_utils import normalize_for_percep
from utils.loss_utils import huber_loss, psnr, ssim


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


def log_first(scene, DeformModel, gaussians, ppt, lpt, background, train_dir, t):
    codedict = {}
    codedict["shape"] = scene.shape_param.to(scene.device)

    viewpoint_cam: Camera = CameraDataset(scene.getTrainCameras(), scene.bg_image)[0]
    viewpoint_cam.to(torch.device(scene.device))

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

    # Render
    render_pkg = render(
        viewpoint_cam,
        gaussians,
        ppt,
        background,
        override_color=None if not lpt.conditional_appearance else gaussians.get_color,
    )

    image = render_pkg["render"]
    gt_image = viewpoint_cam.original_image

    image_np = (image * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    gt_image_np = (gt_image * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    image_pil = Image.fromarray(image_np)
    gt_image_pil = Image.fromarray(gt_image_np)

    image_pil.save(train_dir / f"{t}s_render.png")
    gt_image_pil.save(train_dir / f"{t}s_gt.png")


def log_test_report(scene, DeformModel, gaussians, ppt, lpt, background, percep_module, iteration):
    test_dataloader = DataLoader(
        CameraDataset(scene.getTestCameras(), scene.bg_image),
        batch_size=1,
        shuffle=False,
        num_workers=lpt.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    n_log_images = 8
    log_image_indices = list(
        range(0, len(test_dataloader) - (len(test_dataloader) % n_log_images), len(test_dataloader) // n_log_images)
    )

    log_images = []
    log_gt_images = []
    psnrs = []
    ssims = []
    lpipss = []
    fps = []

    codedict = {}
    codedict["shape"] = scene.shape_param.to(scene.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    i = 0
    for batch in tqdm(test_dataloader, "Testing"):
        viewpoint_cam: Camera = batch[0]
        viewpoint_cam.to(torch.device(scene.device))

        start.record()

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

        # Render
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            ppt,
            background,
            override_color=None if not lpt.conditional_appearance else gaussians.get_color,
        )

        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image

        end.record()
        torch.cuda.synchronize()
        fps.append(1.0 / (start.elapsed_time(end) / 1000))  # CUDA Event measures time in ms

        head_mask = viewpoint_cam.head_mask
        image_percep = normalize_for_percep(image * head_mask)
        gt_image_percep = normalize_for_percep(gt_image * head_mask)

        ssims.append(torch.mean(ssim(image * head_mask, gt_image * head_mask)).item())
        psnrs.append(torch.mean(psnr(image * head_mask, gt_image * head_mask)).item())
        lpipss.append(torch.mean(percep_module.forward(image_percep, gt_image_percep)).item())

        if i in log_image_indices:
            log_images.append(image.cpu())
            log_gt_images.append(gt_image.cpu())

        i += 1

    grid = torchvision.utils.make_grid([*log_images, *log_gt_images], nrow=n_log_images, padding=0)
    wandb.log(
        {
            "test/psnr": np.mean(psnrs),
            "test/ssim": np.mean(ssims),
            "test/lpips": np.mean(lpipss),
            "test/image": [wandb.Image(grid)],
            "test/fps": np.mean(fps),
        },
        step=iteration,
    )

    # Cleanup
    del test_dataloader
    test_dataloader = None
    del log_images
    log_images = None
    del log_gt_images
    log_gt_images = None
    torch.cuda.empty_cache()


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--idname", type=str, default="id1_25", help="id name")
    parser.add_argument("--logname", type=str, default="log", help="log name")
    parser.add_argument("--image_res", type=int, default=512, help="image resolution")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--ours", action="store_true", default=False)
    parser.add_argument("--skip_test", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--skip_wandb", action="store_true", default=False)
    parser.add_argument("--time_log", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    job_name = f"{args.idname}_{args.logname}"
    if not args.skip_wandb:
        wandb.init(project=lpt.project, name=job_name, config={**vars(lpt), **vars(opt), **vars(ppt), **vars(args)})

    batch_size = 1
    set_random_seed(args.seed)

    percep_module = lpips.LPIPS(net="vgg").to(args.device)

    ## deform model
    if args.ours:
        DeformModel = HyperGaussians_Deform_Model(lpt, args.device).to(args.device)
    else:
        DeformModel = Deform_Model(lpt, args.device).to(args.device)

    DeformModel.training_setup()

    ## dataloader
    data_dir = Path("dataset") / args.idname
    mica_datadir = Path("metrical-tracker/output") / args.idname
    log_dir = Path("logs") / job_name
    train_dir = log_dir / "train"
    model_dir = log_dir / "ckpt"

    log_dir.mkdir(exist_ok=True, parents=True)
    train_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    scene = Scene_mica(data_dir, mica_datadir, train_type=0, white_background=lpt.white_background, device=args.device)
    dataloader = DataLoader(
        CameraDataset(scene.getTrainCameras(), scene.bg_image),
        batch_size=batch_size,
        shuffle=True,
        num_workers=lpt.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=lambda x: x,
    )
    dataloader_iter = iter(dataloader)

    # Performance Profiling
    mapping_start = torch.cuda.Event(enable_timing=True)
    mapping_end = torch.cuda.Event(enable_timing=True)
    render_start = torch.cuda.Event(enable_timing=True)
    render_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)

    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    codedict = {}
    codedict["shape"] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    num_params = sum(p.numel() for p in DeformModel.parameters() if p.requires_grad)
    print(f"# Params: {num_params}")
    if not args.skip_wandb:
        wandb.log({"num_params": num_params}, step=0)
        wandb.watch((DeformModel,), log="all", log_freq=1000)

    viewpoint_stack = None
    first_iter += 1
    mid_num = 15000
    epoch_counter = 0
    time_0 = time.time()
    time_1 = time_0
    for iteration in tqdm(range(first_iter, opt.iterations + 1), "Training"):
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getCameras().copy()
        #     random.shuffle(viewpoint_stack)
        #     if len(viewpoint_stack) > 2000:
        #         viewpoint_stack = viewpoint_stack[:2000]
        # viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        try:
            viewpoint_cam: Camera = next(dataloader_iter)[0]  # Batch size is 1
        except StopIteration:
            epoch_counter += 1
            dataloader_iter = iter(dataloader)
            viewpoint_cam: Camera = next(dataloader_iter)[0]  # Batch size is 1

        viewpoint_cam.to(torch.device(args.device))

        frame_id = viewpoint_cam.uid

        # deform gaussians
        if args.profile:
            mapping_start.record()

        codedict["expr"] = viewpoint_cam.exp_param
        codedict["eyes_pose"] = viewpoint_cam.eyes_pose
        codedict["eyelids"] = viewpoint_cam.eyelids
        codedict["jaw_pose"] = viewpoint_cam.jaw_pose
        codedict["camera_center"] = viewpoint_cam.camera_center
        if lpt.conditional_appearance:
            verts_final, rot_delta, scale_coef, uncertainty, color, opacity = DeformModel.decode(codedict)
        else:
            verts_final, rot_delta, scale_coef, uncertainty = DeformModel.decode(codedict)

        if iteration == 1:
            gaussians.create_from_verts(verts_final[0])
            gaussians.training_setup(opt)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        if lpt.conditional_appearance:
            gaussians.update_clr_opc(color[0], opacity[0])

        if args.profile:
            mapping_end.record()

        # Render
        if args.profile:
            render_start.record()
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            ppt,
            background,
            override_color=None if not lpt.conditional_appearance else gaussians.get_color,
        )

        if args.profile:
            render_end.record()

        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image
        mouth_mask = viewpoint_cam.mouth_mask

        loss_huber = huber_loss(image, gt_image, 0.1) + 40 * huber_loss(image * mouth_mask, gt_image * mouth_mask, 0.1)

        loss_G = 0.0
        head_mask = viewpoint_cam.head_mask
        image_percep = normalize_for_percep(image * head_mask)
        gt_image_percep = normalize_for_percep(gt_image * head_mask)
        if iteration > mid_num:
            loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep)) * 0.05

        loss_opacity_reg = 0.0
        if opt.lambda_opacity_reg > 0:
            loss_opacity_reg = ((1 - gaussians.get_opacity) ** 2).mean() * opt.lambda_opacity_reg

        loss = loss_huber * 1 + loss_G * 1 + loss_opacity_reg * 1

        if args.profile:
            backward_start.record()

        loss.backward()

        with torch.no_grad():
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                DeformModel.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                DeformModel.optimizer.zero_grad(set_to_none=True)

            if args.profile:
                backward_end.record()

            # print loss
            if iteration % 500 == 0:
                if iteration <= mid_num:
                    print("step: %d, huber: %.5f" % (iteration, loss_huber.item()))
                else:
                    print("step: %d, huber: %.5f, percep: %.5f" % (iteration, loss_huber.item(), loss_G.item()))

            if not args.skip_wandb:
                wandb.log(
                    {
                        "losses/huber": loss_huber.item(),
                        "losses/percep": 0 if iteration <= mid_num else loss_G.item(),
                        "losses/opacity_reg": (
                            0 if opt.lambda_opacity_reg == 0 else loss_opacity_reg.item() / opt.lambda_opacity_reg
                        ),
                        "epoch": epoch_counter,
                        **(
                            {}
                            if not args.profile
                            else {
                                "profiling/mapping_time": mapping_start.elapsed_time(mapping_end),
                                "profiling/render_time": render_start.elapsed_time(render_end),
                                "profiling/backward_time": backward_start.elapsed_time(backward_end),
                            }
                        ),
                    },
                    step=iteration,
                )

            if iteration % 1250 == 0 and iteration > 0 and not args.skip_test:
                with torch.no_grad():
                    log_test_report(scene, DeformModel, gaussians, ppt, lpt, background, percep_module, iteration)

            # Every 5 seconds for 10 minutes
            if args.time_log and time.time() - time_1 > 5 and time_1 - time_0 < 10 * 60:
                time_1 = time.time()
                log_first(scene, DeformModel, gaussians, ppt, lpt, background, train_dir, int(time_1 - time_0))

            # # visualize results
            # if iteration % 500 == 0 or iteration == 1:
            #     w, h = gt_image.shape[2], gt_image.shape[1]
            #     save_image = np.zeros((h, w * 2, 3))
            #     gt_image_np = (gt_image * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            #     image = image.clamp(0, 1)
            #     image_np = (image * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            #     save_image[:, : w, :] = gt_image_np
            #     save_image[:, w :, :] = image_np
            #     cv2.imwrite(str(train_dir / f"{iteration}.jpg"), save_image[:, :, [2, 1, 0]])

            # save checkpoint
            if iteration % 5000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (DeformModel.capture(), gaussians.capture(), iteration),
                    model_dir / f"ckpt{str(iteration)}.pth",
                )


if __name__ == "__main__":
    main()
