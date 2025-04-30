import json
import math
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov


class Scene_mica:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device, cross=None):
        self.device = device
        self.train_type = train_type
        ## train_type: 0 for train, 1 for test, 2 for eval
        self.frame_delta = 1  # default mica-tracking starts from the second frame
        if cross is not None:
            datadir = os.path.join(datadir, "..", cross)
        self.images_folder = os.path.join(datadir, "imgs")
        self.parsing_folder = os.path.join(datadir, "parsing")
        self.alpha_folder = os.path.join(datadir, "alpha")

        self.bg_image = torch.zeros((3, 512, 512))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        self.mica_ckpt_dir = os.path.join(mica_datadir, "checkpoint")
        if cross is not None:
            self.cross_mica_ckpts = os.path.join(mica_datadir, "..", cross, "checkpoint")
        self.N_frames = len(os.listdir(self.mica_ckpt_dir)) if cross is None else len(os.listdir(self.cross_mica_ckpts))
        self.cameras = []
        test_num = 500
        eval_num = 50
        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(self.mica_ckpt_dir, "00000.frame")
        payload = torch.load(ckpt_path)
        flame_params = payload["flame"]
        self.shape_param = torch.as_tensor(flame_params["shape"])
        orig_w, orig_h = payload["img_size"]
        K = payload["opencv"]["K"][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        self.FovY = focal2fov(fl_y, orig_h)
        self.FovX = focal2fov(fl_x, orig_w)

        train_range_down = 0
        train_range_up = train_num

        test_range_down = self.N_frames - test_num
        test_range_up = self.N_frames

        eval_range_down = self.N_frames - eval_num
        eval_range_up = self.N_frames

        if cross is not None:
            self.mica_ckpt_dir = self.cross_mica_ckpts
        self.train_cameras = self.loadCameras(train_range_down, train_range_up, split="Train")
        self.test_cameras = self.loadCameras(test_range_down, test_range_up, split="Test")
        self.eval_cameras = self.loadCameras(eval_range_down, eval_range_up, split="Eval")

    def loadCameras(self, range_down, range_up, split="Train"):
        cameras = []
        for frame_id in tqdm(range(range_down, range_up), desc=f"Loading {split} Cameras"):
            image_name_mica = str(frame_id).zfill(5)  # obey mica tracking
            image_name_ori = str(frame_id + self.frame_delta).zfill(5)
            ckpt_path = os.path.join(self.mica_ckpt_dir, image_name_mica + ".frame")
            payload = torch.load(ckpt_path)

            flame_params = payload["flame"]
            exp_param = torch.as_tensor(flame_params["exp"])
            eyes_pose = torch.as_tensor(flame_params["eyes"])
            eyelids = torch.as_tensor(flame_params["eyelids"])
            jaw_pose = torch.as_tensor(flame_params["jaw"])

            oepncv = payload["opencv"]
            w2cR = oepncv["R"][0]
            w2cT = oepncv["t"][0]
            R = np.transpose(w2cR)  # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            image_path = os.path.join(self.images_folder, image_name_ori + ".jpg")
            if not os.path.exists(image_path):
                image_path = os.path.join(self.images_folder, image_name_ori + ".png")
            # image = Image.open(image_path)
            # resized_image_rgb = PILtoTensor(image)
            # gt_image = resized_image_rgb[:3, ...]

            # alpha
            alpha_path = os.path.join(self.alpha_folder, image_name_ori + ".jpg")
            if not os.path.exists(alpha_path):
                alpha_path = os.path.join(self.alpha_folder, image_name_ori + ".png")
            # alpha = Image.open(alpha_path)
            # alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(self.parsing_folder, image_name_ori + "_neckhead.png")
            # head_mask = Image.open(head_mask_path)
            # head_mask = PILtoTensor(head_mask)
            # gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            # gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(self.parsing_folder, image_name_ori + "_mouth.png")
            # mouth_mask = Image.open(mouth_mask_path)
            # mouth_mask = PILtoTensor(mouth_mask)

            camera_indiv = Camera(
                colmap_id=frame_id,
                R=R,
                T=T,
                FoVx=self.FovX,
                FoVy=self.FovY,
                image=None,  # gt_image,
                image_path=image_path,
                alpha_path=alpha_path,
                head_mask=None,  # head_mask,
                head_mask_path=head_mask_path,
                mouth_mask=None,  # mouth_mask,
                mouth_mask_path=mouth_mask_path,
                exp_param=exp_param,
                eyes_pose=eyes_pose,
                eyelids=eyelids,
                jaw_pose=jaw_pose,
                image_name=image_name_mica,
                uid=frame_id,
                data_device=self.device,
            )
            cameras.append(camera_indiv)

        return cameras

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

    def getEvalCameras(self):
        return self.eval_cameras

    def getCameras(self):  # Backwards compatibility
        return {0: self.getTrainCameras(), 1: self.getTestCameras(), 2: self.getEvalCameras()}[self.train_type]
