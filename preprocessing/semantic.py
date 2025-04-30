import sys

sys.path.insert(0, "submodules/face_parsing")

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import tyro
from PIL import Image
from submodules.face_parsing.model import BiSeNet
from tqdm import tqdm


def vis_parsing_maps(im, parsing_anno, stride, save_path: Path, save_im=False):
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [255, 255, 0],
        [255, 255, 85],
        [255, 255, 170],
        [255, 0, 255],
        [255, 85, 255],
        [255, 170, 255],
        [0, 255, 255],
        [85, 255, 255],
        [170, 255, 255],
    ]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))
    vis_parsing_anno_color_mouth = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        if pi == 16:  # Shoulders
            continue
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = 255  # part_colors[pi]

        if pi in [11, 12, 13]:  # Lower Lip, Upper Lip, Mouth Interior
            vis_parsing_anno_color_mouth[index[0], index[1], :] = 255

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_parsing_anno_color_mouth = vis_parsing_anno_color_mouth.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(str(save_path.parent / f"{save_path.stem}_neckhead.png"), vis_parsing_anno_color)
        cv2.imwrite(str(save_path.parent / f"{save_path.stem}_mouth.png"), vis_parsing_anno_color_mouth)
        # cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im


def main(input_path: Path, output_path: Path = None, checkpoint_path: Path = None):
    print(str(input_path))
    if output_path is None:
        output_path = input_path.parent / "parsing"
    output_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = Path("submodules/face_parsing/res/cp/79999_iter.pth")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Download the pre-trained model from https://github.com/zllrunning/face-parsing.PyTorch"
        )

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        images = list(input_path.iterdir())
        for image_path in tqdm(images, desc="Parsing Faces"):
            image = Image.open(image_path)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_path=output_path / image_path.name, save_im=True)


if __name__ == "__main__":
    tyro.cli(main)
