from copy import deepcopy

import torch
from PIL import Image
from torch.utils.data import Dataset

from scene.cameras import Camera
from utils.general_utils import PILtoTensor


class CameraDataset(Dataset):
    def __init__(self, cameras, bg_image, cross=False):
        self.cameras = cameras
        self.bg_image: torch.Tensor = bg_image
        self.cross = cross

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return CameraDataset(self.cameras[idx], self.bg_image)

        if not isinstance(idx, int):
            raise TypeError(f"Invalid argument type: {type(idx)}")

        camera: Camera = deepcopy(self.cameras[idx])

        try:
            image = Image.open(camera.image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]

            alpha = Image.open(camera.alpha_path)
            alpha = PILtoTensor(alpha)

            head_mask = Image.open(camera.head_mask_path)
            head_mask = PILtoTensor(head_mask)
            if self.bg_image.shape[1] != gt_image.shape[1] or self.bg_image.shape[2] != gt_image.shape[2]:
                self.bg_image = torch.ones_like(gt_image)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            mouth_mask = Image.open(camera.mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)

            camera.original_image = gt_image.clamp(0.0, 1.0)
            camera.head_mask = head_mask
            camera.mouth_mask = mouth_mask

            camera.image_width = camera.original_image.shape[2]
            camera.image_height = camera.original_image.shape[1]
        except Exception as e:
            if not self.cross:
                raise e

        return camera
