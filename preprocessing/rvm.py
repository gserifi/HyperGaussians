from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image
from tqdm import tqdm


def main(video_path: Path):
    print(str(video_path))
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
    model.to("cuda")

    alpha_path = video_path.parent / "alpha"
    alpha_path.mkdir(exist_ok=True)

    comp_path = video_path.parent / "comp"
    comp_path.mkdir(exist_ok=True)

    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
    convert_video(
        model,
        input_source=video_path,
        downsample_ratio=1.0,
        output_type="png_sequence",
        output_alpha=alpha_path,
        output_composition=comp_path,
    )

    # Convert transparent to white background
    frames = list(comp_path.iterdir())
    for frame in tqdm(frames[::-1]):
        im = Image.open(frame)
        im = np.array(im)[..., :3] / 255.0

        mask = Image.open(alpha_path / frame.name)
        mask = (np.array(mask) / 255.0)[..., None]

        im = im * mask + (1 - mask)

        im = Image.fromarray((im * 255).astype(np.uint8))
        im.save(frame)


if __name__ == "__main__":
    tyro.cli(main)
