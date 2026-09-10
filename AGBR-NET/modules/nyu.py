"""
Downloads the NYU Depth V2 labeled dataset (via Hugging Face's hosted
copy) and saves it into the rgb/ + depth/ folder structure train_gan.py
expects.

Run this once, on whatever machine you'll train on (Colab recommended,
since it also needs a GPU for train_gan.py itself).

Install requirements first:
    pip install datasets pillow numpy

Usage:
    python download_nyu_data.py --output_dir ./data --limit 400

--limit caps how many image pairs to save. You do NOT need the full 1449
for a first working model — a few hundred is plenty to get train_gan.py
running end-to-end and producing a usable checkpoint before December.
Increase it later once you know the pipeline works.
"""

import os
import argparse
import numpy as np
from PIL import Image


def main(output_dir, limit):
    from datasets import load_dataset  # imported here so --help works without the package installed

    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    print("Downloading NYU Depth V2 from Hugging Face (this can take a while the first time)...")
    dataset = load_dataset("jagennath-hari/nyuv2", split="train")

    n_total = len(dataset) if limit is None else min(limit, len(dataset))
    print(f"Saving {n_total} RGB/depth pairs to {output_dir} ...")

    for i in range(n_total):
        sample = dataset[i]

        rgb_image = sample["rgb"]  # PIL Image, RGB
        depth_image = sample["depth"]  # PIL Image (tiff, uint16, millimeters)

        filename = f"{i:05d}.png"

        rgb_image.convert("RGB").save(os.path.join(rgb_dir, filename))

        # Depth may come back as a PIL image or a raw array depending on
        # dataset version — handle both, always save as a 16-bit PNG so
        # precision isn't lost (matches what load_depth() in train_gan.py
        # expects: cv2.IMREAD_UNCHANGED on a single-channel image).
        if isinstance(depth_image, Image.Image):
            depth_arr = np.array(depth_image).astype(np.float32)
        else:
            depth_arr = np.array(depth_image).astype(np.float32)

        # Normalize to a 16-bit range for storage (preserves relative
        # depth ordering, which is all train_gan.py's normalize_patch()
        # needs — it re-normalizes per-patch anyway).
        d_min, d_max = depth_arr.min(), depth_arr.max()
        denom = d_max - d_min
        if denom < 1e-6:
            depth_16bit = np.zeros_like(depth_arr, dtype=np.uint16)
        else:
            depth_16bit = ((depth_arr - d_min) / denom * 65535).astype(np.uint16)

        Image.fromarray(depth_16bit).save(os.path.join(depth_dir, filename))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_total} saved...")

    print(f"Done. {n_total} pairs saved to:")
    print(f"  {rgb_dir}")
    print(f"  {depth_dir}")
    print(f"\nNow run: python train_gan.py --data_dir \"{output_dir}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--limit", type=int, default=400,
                         help="Number of image pairs to download. Default 400 is enough "
                              "to get training running; use a larger number (or omit for all "
                              "1449) once you've confirmed the pipeline works end-to-end.")
    args = parser.parse_args()

    main(args.output_dir, args.limit)