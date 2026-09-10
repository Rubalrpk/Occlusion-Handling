"""
Boundary Refinement GAN — training script.

RUN THIS ON COLAB (or any machine with a GPU), NOT on your laptop.
This is a completely separate concern from the live CPU pipeline you've
been testing — training happens once, offline; the exported ONNX model
is what actually runs live.

--------------------------------------------------------------------------
WHY SYNTHETIC COARSE/SHARP PAIRS INSTEAD OF REAL OCCLUSION GROUND TRUTH
--------------------------------------------------------------------------
You don't have (and don't need) hand-labeled occlusion boundary ground
truth to train this. Instead: take a dataset of clean, sharp depth maps
(NYU Depth V2 is the standard, widely-available choice), and manufacture
a "coarse" version of each by downsampling then upsampling it — this
simulates the boundary blur MiDaS actually produces near real occlusion
edges. The generator then learns coarse -> sharp, which is exactly the
mapping it needs to perform at inference time on real MiDaS output.

--------------------------------------------------------------------------
EXPECTED DATA FORMAT
--------------------------------------------------------------------------
A directory of paired files:
    data/
      rgb/    0001.png, 0002.png, ...
      depth/  0001.png, 0002.png, ...   (or .npy / .pfm — see load_depth())
with matching filenames between rgb/ and depth/. Adjust load_depth() below
if your dataset's depth format differs (NYU Depth V2's .mat format needs
a small loader change — not included here since export format varies by
how you download it).

--------------------------------------------------------------------------
WHY PATCHES ARE SAMPLED NEAR EDGES, NOT RANDOMLY
--------------------------------------------------------------------------
region_selector.py only ever sends the GAN patches from low-confidence
(i.e. high-gradient / boundary) regions — never smooth interior regions.
If this script trained on uniformly random patches, most training patches
would be boring flat regions the GAN will never actually see in
deployment. Training patches are instead sampled preferentially from
high-gradient locations in the sharp depth map, to match the real
inference-time distribution.
"""

import os
import glob
import random
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from gan_architecture import BoundaryRefinementGenerator, PatchDiscriminator, PATCH_SIZE


def load_depth(path):
    """Load a depth map. Default assumes single-channel PNG/EXR readable
    by cv2. Change this if your dataset stores depth differently
    (e.g. .npy arrays, NYU's .mat format)."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        # cv2.imread returns None both when the file is missing AND when
        # it exists but fails to decode (truncated/corrupted PNG, etc).
        # Distinguish the two so the real cause is obvious.
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Depth file does not exist: {path}")
        else:
            raise ValueError(
                f"Depth file exists but failed to decode (likely corrupted/"
                f"truncated — possibly from an interrupted download): {path}"
            )
    return depth.astype(np.float32)


def normalize_patch(patch):
    """Per-patch min-max normalization to [0,1] — matches what
    gan_refinement.py must do at inference time. Keep these two
    normalization schemes IDENTICAL or the trained model will see a
    different input distribution live than it saw during training."""
    p_min, p_max = patch.min(), patch.max()
    denom = p_max - p_min
    if denom < 1e-6:
        return np.zeros_like(patch, dtype=np.float32), (p_min, p_max)
    return (patch - p_min) / denom, (p_min, p_max)


def find_edge_biased_center(sharp_depth, patch_size, edge_bias=0.8):
    """Pick a patch center, biased toward high-gradient (boundary)
    locations in the depth map — matches the distribution region_selector
    will actually send at inference time."""
    h, w = sharp_depth.shape[:2]
    half = patch_size // 2

    if random.random() < edge_bias:
        gx = cv2.Sobel(sharp_depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sharp_depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        # Restrict to region where a full patch fits inside the image
        valid = grad_mag[half:h - half, half:w - half]
        if valid.size > 0 and valid.max() > 1e-6:
            flat_idx = np.argmax(valid.flatten() * np.random.rand(valid.size))
            # ^ weighted-random pick among high-gradient pixels, not just the single max
            y, x = np.unravel_index(flat_idx, valid.shape)
            return x + half, y + half

    # Fallback: uniform random center
    return random.randint(half, w - half - 1), random.randint(half, h - half - 1)


class CoarseSharpDepthDataset(Dataset):
    def __init__(self, data_dir, patch_size=PATCH_SIZE, degrade_factor=0.25, samples_per_image=8):
        self.rgb_paths = sorted(glob.glob(os.path.join(data_dir, "rgb", "*")))
        self.depth_dir = os.path.join(data_dir, "depth")
        self.patch_size = patch_size
        self.degrade_factor = degrade_factor
        self.samples_per_image = samples_per_image

        if not self.rgb_paths:
            raise FileNotFoundError(
                f"No RGB images found in {os.path.join(data_dir, 'rgb')}. "
                "Check DATA_DIR and folder structure (see module docstring)."
            )

    def __len__(self):
        return len(self.rgb_paths) * self.samples_per_image

    def __getitem__(self, idx):
        # Retry a few times with different samples if one turns out to be
        # corrupted — a handful of bad files shouldn't crash an entire
        # training run. If it keeps failing, something more serious is
        # wrong and we let the real error surface.
        max_retries = 5
        last_error = None

        for attempt in range(max_retries):
            try_idx = idx if attempt == 0 else random.randrange(len(self))
            img_idx = try_idx // self.samples_per_image
            rgb_path = self.rgb_paths[img_idx]
            depth_path = os.path.join(self.depth_dir, os.path.basename(rgb_path))

            try:
                rgb = cv2.imread(rgb_path)
                if rgb is None:
                    raise ValueError(f"Failed to decode RGB file: {rgb_path}")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                sharp_depth = load_depth(depth_path)
                return self._build_sample(rgb, sharp_depth)
            except (FileNotFoundError, ValueError, cv2.error) as e:
                last_error = e
                print(f"WARNING: skipping corrupted sample (index {img_idx}, "
                      f"{os.path.basename(rgb_path)}): {e}")
                continue

        raise RuntimeError(
            f"Failed to load a valid sample after {max_retries} retries. "
            f"Last error: {last_error}. This suggests more than a couple of "
            f"stray corrupted files — worth running a full dataset check."
        )

    def _build_sample(self, rgb, sharp_depth):
        h, w = sharp_depth.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            # Image smaller than a patch — resize up rather than crash.
            rgb = cv2.resize(rgb, (max(w, ps), max(h, ps)))
            sharp_depth = cv2.resize(sharp_depth, (max(w, ps), max(h, ps)))
            h, w = sharp_depth.shape[:2]

        cx, cy = find_edge_biased_center(sharp_depth, ps)
        x0, y0 = cx - ps // 2, cy - ps // 2

        rgb_patch = rgb[y0:y0 + ps, x0:x0 + ps]
        sharp_patch = sharp_depth[y0:y0 + ps, x0:x0 + ps]

        # Manufacture the coarse (blurred/boundary-degraded) version.
        small_size = max(1, int(ps * self.degrade_factor))
        coarse_patch = cv2.resize(sharp_patch, (small_size, small_size), interpolation=cv2.INTER_AREA)
        coarse_patch = cv2.resize(coarse_patch, (ps, ps), interpolation=cv2.INTER_LINEAR)

        sharp_norm, _ = normalize_patch(sharp_patch)
        coarse_norm, _ = normalize_patch(coarse_patch)

        rgb_tensor = torch.from_numpy(rgb_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        coarse_tensor = torch.from_numpy(coarse_norm).unsqueeze(0)
        sharp_tensor = torch.from_numpy(sharp_norm).unsqueeze(0)

        input_tensor = torch.cat([rgb_tensor, coarse_tensor], dim=0)  # 4 channels

        return input_tensor, sharp_tensor


def gradient_loss(pred, target):
    """L1 loss on spatial gradients — pushes the generator toward sharp
    boundaries specifically, not just pixel-wise correctness. Directly
    supports the 'boundary-guided' framing of the paper."""
    def grad(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    pred_dx, pred_dy = grad(pred)
    target_dx, target_dy = grad(target)
    return nn.functional.l1_loss(pred_dx, target_dx) + nn.functional.l1_loss(pred_dy, target_dy)


def train(data_dir, output_dir, epochs=30, batch_size=16, lr=2e-4,
          lambda_l1=100.0, lambda_grad=50.0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. This will be extremely slow — "
              "you almost certainly want to be running this on Colab, not locally.")

    os.makedirs(output_dir, exist_ok=True)

    dataset = CoarseSharpDepthDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    gen = BoundaryRefinementGenerator().to(device)
    disc = PatchDiscriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    adv_loss_fn = nn.MSELoss()  # LSGAN-style, more stable than vanilla BCE for small datasets
    l1_loss_fn = nn.L1Loss()

    for epoch in range(epochs):
        running_g, running_d = 0.0, 0.0

        for input_tensor, sharp_tensor in loader:
            input_tensor = input_tensor.to(device)
            sharp_tensor = sharp_tensor.to(device)
            rgb = input_tensor[:, :3]

            # --- Train discriminator ---
            opt_d.zero_grad()
            fake_depth = gen(input_tensor).detach()

            real_pair = torch.cat([rgb, sharp_tensor], dim=1)
            fake_pair = torch.cat([rgb, fake_depth], dim=1)

            real_pred = disc(real_pair)
            fake_pred = disc(fake_pair)

            d_loss = 0.5 * (
                adv_loss_fn(real_pred, torch.ones_like(real_pred)) +
                adv_loss_fn(fake_pred, torch.zeros_like(fake_pred))
            )
            d_loss.backward()
            opt_d.step()

            # --- Train generator ---
            opt_g.zero_grad()
            fake_depth = gen(input_tensor)
            fake_pair = torch.cat([rgb, fake_depth], dim=1)
            fake_pred = disc(fake_pair)

            g_adv = adv_loss_fn(fake_pred, torch.ones_like(fake_pred))
            g_l1 = l1_loss_fn(fake_depth, sharp_tensor)
            g_grad = gradient_loss(fake_depth, sharp_tensor)

            g_loss = g_adv + lambda_l1 * g_l1 + lambda_grad * g_grad
            g_loss.backward()
            opt_g.step()

            running_g += g_loss.item()
            running_d += d_loss.item()

        print(f"Epoch {epoch+1}/{epochs}  G_loss={running_g/len(loader):.4f}  D_loss={running_d/len(loader):.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(output_dir, f"generator_epoch{epoch+1}.pt")
            torch.save(gen.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final export to ONNX for CPU inference in the live pipeline.
    export_onnx(gen, os.path.join(output_dir, "boundary_refinement_gan.onnx"), device)


def export_onnx(generator, output_path, device):
    generator.eval()
    dummy_input = torch.randn(1, 4, PATCH_SIZE, PATCH_SIZE).to(device)
    torch.onnx.export(
        generator, dummy_input, output_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
        dynamo=False,  # forces a single self-contained .onnx file (no
                        # external .onnx.data companion file to lose track of)
    )
    print(f"Exported ONNX model to: {output_path}")
    print("Copy this file to your project's models/ folder and point "
          "gan_refinement.py's DEFAULT_MODEL_PATH at it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing rgb/ and depth/ subfolders (see module docstring)")
    parser.add_argument("--output_dir", type=str, default="./gan_checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train(args.data_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size)