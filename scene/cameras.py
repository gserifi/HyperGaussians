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

import numpy as np
import torch
from torch import nn

from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera(nn.Module):

    world_view_transform: torch.Tensor
    projection_matrix: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor

    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        image_path,
        alpha_path,
        head_mask,
        head_mask_path,
        mouth_mask,
        mouth_mask_path,
        exp_param,
        eyes_pose,
        eyelids,
        jaw_pose,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_path = image_path
        self.alpha_path = alpha_path
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0)  # .to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            self.image_width = 512
            self.image_height = 512

        self.head_mask_path = head_mask_path
        if head_mask is not None:
            self.head_mask = head_mask  # .to(self.data_device)

        self.mouth_mask_path = mouth_mask_path
        if mouth_mask is not None:
            self.mouth_mask = mouth_mask  # .to(self.data_device)

        self.exp_param = exp_param  # .to(self.data_device)
        self.eyes_pose = eyes_pose  # .to(self.data_device)
        self.eyelids = eyelids  # .to(self.data_device)
        self.jaw_pose = jaw_pose  # .to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.compute_transforms(self.R, self.T)

    def compute_transforms(self, rotation: torch.Tensor, translation: torch.Tensor):
        self.world_view_transform = torch.tensor(
            getWorld2View2(rotation, translation, self.trans, self.scale)
        ).transpose(
            0, 1
        )  # .cuda()
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(
            0, 1
        )  # .cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def to(self, device: torch.device):
        try:
            self.original_image = self.original_image.to(device)
            self.head_mask = self.head_mask.to(device)
            self.mouth_mask = self.mouth_mask.to(device)
        except AttributeError:
            pass

        self.exp_param = self.exp_param.to(device)
        self.eyes_pose = self.eyes_pose.to(device)
        self.eyelids = self.eyelids.to(device)
        self.jaw_pose = self.jaw_pose.to(device)

        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
