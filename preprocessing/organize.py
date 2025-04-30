from pathlib import Path

import tyro


def main(data_path: Path):
    print(str(data_path))
    # FlashAvatar expects all files to be named with a 5-digit frame number. Some of the preprocessing steps produce different filenames.

    alpha_path = data_path / "alpha"
    imgs_path = data_path / "imgs"
    parsing_path = data_path / "parsing"

    if not imgs_path.exists() and (data_path / "comp").exists():
        (data_path / "comp").rename(imgs_path)

    if not alpha_path.exists():
        raise FileNotFoundError(f"Directory {alpha_path} does not exist. Run preprocessing/rvm.py first.")

    if not imgs_path.exists():
        raise FileNotFoundError(f"Directory {imgs_path} does not exist. Run preprocessing/rvm.py first.")

    if not parsing_path.exists():
        raise FileNotFoundError(f"Directory {parsing_path} does not exist. Run preprocessing/semantic.py first.")

    # Filenames must match temporal order
    alphas = list(sorted(alpha_path.iterdir()))
    imgs = list(sorted(imgs_path.iterdir()))
    parsings = list(sorted(parsing_path.iterdir()))
    neckhead_parsings = [x for x in parsings if "neckhead" in x.stem]
    mouth_parsings = [x for x in parsings if "mouth" in x.stem]

    for frame_id, (alpha, img, neckhead_parsing, mouth_parsing) in enumerate(
        zip(alphas, imgs, neckhead_parsings, mouth_parsings)
    ):
        alpha.rename(alpha_path / f"{frame_id:05d}.png")
        img.rename(imgs_path / f"{frame_id:05d}.png")
        neckhead_parsing.rename(parsing_path / f"{frame_id:05d}_neckhead.png")
        mouth_parsing.rename(parsing_path / f"{frame_id:05d}_mouth.png")


if __name__ == "__main__":
    tyro.cli(main)
