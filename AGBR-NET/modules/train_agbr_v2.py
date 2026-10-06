import os
import glob
import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from modules.depth_estimation import DepthEstimator


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET"

DATASET_DIR = r"C:\Users\rubal\OneDrive\Desktop\urf"

MIDAS_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "midas",
    "midas_v21_small_256.onnx"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "models",
    "gan_v2"
)

SCENES = [
    "scene_a",
    "scene_b",
    "scene_c",
    "scene_d",
    "scene_e",
    "scene_f",
    "scene_g",
]

# Scene-wise split.
# These scenes are NEVER mixed between train/validation/test.
TRAIN_SCENES = [
    "scene_a",
    "scene_b",
    "scene_c",
    "scene_d",
]

VAL_SCENES = [
    "scene_e",
]

TEST_SCENES = [
    "scene_f",
    "scene_g",
]

PATCH_SIZE = 64

SAMPLES_PER_FRAME = 8

BATCH_SIZE = 16

EPOCHS = 40

LEARNING_RATE = 1e-4

# Loss weights
LAMBDA_DEPTH = 10.0
LAMBDA_BOUNDARY = 15.0
LAMBDA_RESIDUAL = 1.0
LAMBDA_ADV = 0.1

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# UTILITY
# ============================================================

def ensure_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


# ============================================================
# DEPTH NORMALIZATION
# ============================================================

def normalize_with_coarse_range(
    coarse,
    target
):
    """
    IMPORTANT:

    Both coarse MiDaS depth and LiDAR target are represented
    using the SAME normalization range.

    This fixes the previous mismatch where coarse and sharp
    depths were independently min-max normalized.
    """

    c_min = float(
        np.percentile(coarse, 1)
    )

    c_max = float(
        np.percentile(coarse, 99)
    )

    if c_max - c_min < 1e-6:

        coarse_norm = np.zeros_like(
            coarse,
            dtype=np.float32
        )

        target_norm = np.zeros_like(
            target,
            dtype=np.float32
        )

        return (
            coarse_norm,
            target_norm
        )

    coarse_norm = (
        (coarse - c_min)
        /
        (c_max - c_min)
    )

    target_norm = (
        (target - c_min)
        /
        (c_max - c_min)
    )

    coarse_norm = np.clip(
        coarse_norm,
        0.0,
        1.0
    )

    target_norm = np.clip(
        target_norm,
        0.0,
        1.0
    )

    return (
        coarse_norm.astype(np.float32),
        target_norm.astype(np.float32)
    )


# ============================================================
# RESIDUAL GENERATOR
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm=True
    ):

        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1
            )
        ]

        if norm:

            layers.append(
                nn.InstanceNorm2d(
                    out_channels
                )
            )

        layers.append(
            nn.LeakyReLU(
                0.2,
                inplace=True
            )
        )

        self.block = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.block(x)


class DeconvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=False
    ):

        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(
                out_channels
            )
        ]

        if dropout:

            layers.append(
                nn.Dropout(0.2)
            )

        layers.append(
            nn.ReLU(
                inplace=True
            )
        )

        self.block = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.block(x)


class ResidualBoundaryGenerator(
    nn.Module
):

    """
    Input:
        RGB + coarse depth + confidence
        = 5 channels

    Output:
        refined depth

    Instead of predicting an entirely new depth map,
    the network predicts a residual correction:

        refined = coarse + residual
    """

    def __init__(
        self,
        base_channels=32
    ):

        super().__init__()

        c = base_channels

        self.enc1 = ConvBlock(
            5,
            c,
            norm=False
        )

        self.enc2 = ConvBlock(
            c,
            c * 2
        )

        self.enc3 = ConvBlock(
            c * 2,
            c * 4
        )

        self.enc4 = ConvBlock(
            c * 4,
            c * 8
        )

        self.bottleneck = ConvBlock(
            c * 8,
            c * 8,
            norm=False
        )

        self.dec4 = DeconvBlock(
            c * 8,
            c * 8,
            dropout=True
        )

        self.dec3 = DeconvBlock(
            c * 16,
            c * 4
        )

        self.dec2 = DeconvBlock(
            c * 8,
            c * 2
        )

        self.dec1 = DeconvBlock(
            c * 4,
            c
        )

        self.final = nn.ConvTranspose2d(
            c * 2,
            1,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(
        self,
        x
    ):

        coarse = x[:, 3:4]

        e1 = self.enc1(x)

        e2 = self.enc2(e1)

        e3 = self.enc3(e2)

        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b)

        d3 = self.dec3(
            torch.cat(
                [d4, e4],
                dim=1
            )
        )

        d2 = self.dec2(
            torch.cat(
                [d3, e3],
                dim=1
            )
        )

        d1 = self.dec1(
            torch.cat(
                [d2, e2],
                dim=1
            )
        )

        residual = self.final(
            torch.cat(
                [d1, e1],
                dim=1
            )
        )

        # Small residual is encouraged.
        # Tanh limits extreme corrections.
        residual = (
            0.25
            * torch.tanh(
                residual
            )
        )

        refined = (
            coarse
            + residual
        )

        refined = torch.clamp(
            refined,
            0.0,
            1.0
        )

        return refined


# ============================================================
# PATCH DISCRIMINATOR
# ============================================================

class PatchDiscriminator(
    nn.Module
):

    def __init__(
        self,
        base_channels=32
    ):

        super().__init__()

        c = base_channels

        self.model = nn.Sequential(

            ConvBlock(
                4,
                c,
                norm=False
            ),

            ConvBlock(
                c,
                c * 2
            ),

            ConvBlock(
                c * 2,
                c * 4
            ),

            nn.Conv2d(
                c * 4,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(
        self,
        x
    ):

        return self.model(x)


# ============================================================
# BOUNDARY EXTRACTION
# ============================================================

def compute_boundary_weight(
    depth
):

    """
    Creates a boundary emphasis map.

    Large depth gradients receive larger weights.
    """

    dx = torch.abs(
        depth[:, :, :, 1:]
        -
        depth[:, :, :, :-1]
    )

    dy = torch.abs(
        depth[:, :, 1:, :]
        -
        depth[:, :, :-1, :]
    )

    dx = F.pad(
        dx,
        (0, 1, 0, 0)
    )

    dy = F.pad(
        dy,
        (0, 0, 0, 1)
    )

    gradient = (
        dx + dy
    )

    # Normalize each image independently.
    max_value = (
        gradient.amax(
            dim=(2, 3),
            keepdim=True
        )
        + 1e-6
    )

    gradient = (
        gradient
        /
        max_value
    )

    # Base weight = 1
    # Boundary weight can reach 6
    weight = (
        1.0
        +
        5.0 * gradient
    )

    return weight


# ============================================================
# DEPTH DATASET
# ============================================================

class RealPolycamDataset(
    Dataset
):

    def __init__(
        self,
        dataset_root,
        scenes,
        depth_model,
        samples_per_frame=8
    ):

        self.dataset_root = dataset_root

        self.scenes = scenes

        self.depth_model = depth_model

        self.samples_per_frame = (
            samples_per_frame
        )

        self.frames = []

        # Cache MiDaS predictions.
        self.cache_dir = os.path.join(
            OUTPUT_DIR,
            "midas_cache"
        )

        ensure_dir(
            self.cache_dir
        )

        print("\nBuilding dataset...")

        for scene in scenes:

            scene_dir = os.path.join(
                dataset_root,
                scene
            )

            rgb_dir = os.path.join(
                scene_dir,
                "keyframes",
                "images"
            )

            lidar_dir = os.path.join(
                scene_dir,
                "keyframes",
                "depth"
            )

            if not os.path.isdir(
                rgb_dir
            ):

                print(
                    f"WARNING: missing RGB: "
                    f"{rgb_dir}"
                )

                continue

            if not os.path.isdir(
                lidar_dir
            ):

                print(
                    f"WARNING: missing depth: "
                    f"{lidar_dir}"
                )

                continue

            rgb_files = {
                os.path.splitext(
                    f
                )[0]: f

                for f in os.listdir(
                    rgb_dir
                )

                if f.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                )
            }

            depth_files = {
                os.path.splitext(
                    f
                )[0]: f

                for f in os.listdir(
                    lidar_dir
                )

                if f.lower().endswith(
                    ".png"
                )
            }

            ids = sorted(
                set(
                    rgb_files
                )
                &
                set(
                    depth_files
                )
            )

            print(
                f"{scene}: "
                f"{len(ids)} matched frames"
            )

            for frame_id in ids:

                self.frames.append({

                    "scene":
                        scene,

                    "frame_id":
                        frame_id,

                    "rgb":
                        os.path.join(
                            rgb_dir,
                            rgb_files[
                                frame_id
                            ]
                        ),

                    "depth":
                        os.path.join(
                            lidar_dir,
                            depth_files[
                                frame_id
                            ]
                        ),
                })

        if not self.frames:

            raise RuntimeError(
                "No matched RGB/LiDAR frames found."
            )

        print(
            f"Total frames: "
            f"{len(self.frames)}"
        )

    def __len__(self):

        return (
            len(self.frames)
            *
            self.samples_per_frame
        )

    def __getitem__(
        self,
        index
    ):

        frame_index = (
            index
            //
            self.samples_per_frame
        )

        item = self.frames[
            frame_index
        ]

        rgb = cv2.imread(
            item["rgb"]
        )

        if rgb is None:

            raise RuntimeError(
                f"Could not read "
                f"{item['rgb']}"
            )

        rgb = cv2.cvtColor(
            rgb,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # LiDAR
        # ----------------------------------------------------

        lidar_raw = cv2.imread(
            item["depth"],
            cv2.IMREAD_UNCHANGED
        )

        if lidar_raw is None:

            raise RuntimeError(
                f"Could not read "
                f"{item['depth']}"
            )

        lidar = (
            lidar_raw.astype(
                np.float32
            )
            / 1000.0
        )

        # ----------------------------------------------------
        # MiDaS
        # ----------------------------------------------------

        cache_name = (
            f"{item['scene']}_"
            f"{item['frame_id']}.npy"
        )

        cache_path = os.path.join(
            self.cache_dir,
            cache_name
        )

        if os.path.isfile(
            cache_path
        ):

            midas = np.load(
                cache_path
            )

        else:

            midas = (
                self.depth_model.predict(
                    cv2.cvtColor(
                        rgb,
                        cv2.COLOR_RGB2BGR
                    )
                )
            )

            np.save(
                cache_path,
                midas
            )

        # ----------------------------------------------------
        # Resize MiDaS to RGB resolution
        # ----------------------------------------------------

        midas = cv2.resize(
            midas,
            (
                rgb.shape[1],
                rgb.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Resize LiDAR to RGB resolution
        #
        # This is only for extracting aligned training
        # patches. Evaluation remains at native LiDAR
        # resolution.
        # ----------------------------------------------------

        lidar_rgb = cv2.resize(
            lidar,
            (
                rgb.shape[1],
                rgb.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Valid LiDAR mask
        # ----------------------------------------------------

        valid = (
            lidar_rgb > 0
        )

        # ----------------------------------------------------
        # Pick boundary-biased patch
        # ----------------------------------------------------

        lidar_for_gradient = (
            lidar_rgb.copy()
        )

        lidar_for_gradient[
            ~valid
        ] = 0

        gx = cv2.Sobel(
            lidar_for_gradient,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            lidar_for_gradient,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        gradient = np.sqrt(
            gx ** 2
            +
            gy ** 2
        )

        h, w = lidar.shape

        ps = PATCH_SIZE

        # ----------------------------------------------------
        # Candidate centers
        # ----------------------------------------------------

        margin = ps // 2 + 2

        if (
            w > 2 * margin
            and h > 2 * margin
        ):

            candidate = gradient[
                margin:
                rgb.shape[0] - margin,

                margin:
                rgb.shape[1] - margin
            ]

            if candidate.size > 0:

                # 70% boundary sampling
                if random.random() < 0.7:

                    flat = candidate.flatten()

                    # top 20% gradient locations
                    threshold = np.percentile(
                        flat,
                        80
                    )

                    ys, xs = np.where(
                        candidate >= threshold
                    )

                    if len(xs) > 0:

                        choice = random.randrange(
                            len(xs)
                        )

                        cy = (
                            ys[choice]
                            + margin
                        )

                        cx = (
                            xs[choice]
                            + margin
                        )

                    else:

                        cx = random.randint(
                            margin,
                            rgb.shape[1]
                            - margin
                            - 1
                        )

                        cy = random.randint(
                            margin,
                            rgb.shape[0]
                            - margin
                            - 1
                        )

                else:

                    cx = random.randint(
                        margin,
                        rgb.shape[1]
                        - margin
                        - 1
                    )

                    cy = random.randint(
                        margin,
                        rgb.shape[0]
                        - margin
                        - 1
                    )

            else:

                cx = rgb.shape[1] // 2
                cy = rgb.shape[0] // 2

        else:

            cx = rgb.shape[1] // 2
            cy = rgb.shape[0] // 2

        # ----------------------------------------------------
        # Patch coordinates
        # ----------------------------------------------------

        x0 = max(
            0,
            min(
                cx - ps // 2,
                rgb.shape[1] - ps
            )
        )

        y0 = max(
            0,
            min(
                cy - ps // 2,
                rgb.shape[0] - ps
            )
        )

        x1 = x0 + ps
        y1 = y0 + ps

        rgb_patch = rgb[
            y0:y1,
            x0:x1
        ]

        coarse_patch = midas[
            y0:y1,
            x0:x1
        ]

        target_patch = lidar_rgb[
            y0:y1,
            x0:x1
        ]

        valid_patch = valid[
            y0:y1,
            x0:x1
        ]

        # ----------------------------------------------------
        # If insufficient LiDAR coverage,
        # choose another patch recursively.
        # ----------------------------------------------------

        valid_ratio = (
            valid_patch.mean()
        )

        if valid_ratio < 0.50:

            new_index = random.randrange(
                len(self)
            )

            return self[
                new_index
            ]

        # ----------------------------------------------------
        # Same normalization range
        # ----------------------------------------------------

        coarse_norm, target_norm = (
            normalize_with_coarse_range(
                coarse_patch,
                target_patch
            )
        )

        # ----------------------------------------------------
        # Confidence proxy
        #
        # High confidence:
        # smooth MiDaS
        #
        # Low confidence:
        # strong gradient/disagreement
        #
        # This is used as an additional GAN input.
        # ----------------------------------------------------

        mgx = cv2.Sobel(
            coarse_patch,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        mgy = cv2.Sobel(
            coarse_patch,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        mg = np.sqrt(
            mgx ** 2
            +
            mgy ** 2
        )

        mg = mg / (
            mg.max()
            + 1e-6
        )

        confidence = (
            1.0 - mg
        )

        confidence = np.clip(
            confidence,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Tensors
        # ----------------------------------------------------

        rgb_tensor = (
            torch.from_numpy(
                rgb_patch.astype(
                    np.float32
                )
                / 255.0
            )
            .permute(
                2,
                0,
                1
            )
        )

        coarse_tensor = (
            torch.from_numpy(
                coarse_norm
            )
            .unsqueeze(0)
        )

        confidence_tensor = (
            torch.from_numpy(
                confidence.astype(
                    np.float32
                )
            )
            .unsqueeze(0)
        )

        target_tensor = (
            torch.from_numpy(
                target_norm
            )
            .unsqueeze(0)
        )

        input_tensor = torch.cat(
            [
                rgb_tensor,
                coarse_tensor,
                confidence_tensor
            ],
            dim=0
        )

        return (
            input_tensor.float(),
            target_tensor.float()
        )


# ============================================================
# LOSSES
# ============================================================

def boundary_weighted_loss(
    prediction,
    target
):

    weight = compute_boundary_weight(
        target
    )

    error = torch.abs(
        prediction
        -
        target
    )

    return torch.mean(
        weight
        *
        error
    )


def residual_loss(
    prediction,
    coarse
):

    residual = (
        prediction
        -
        coarse
    )

    return torch.mean(
        torch.abs(
            residual
        )
    )


def gradient_loss(
    prediction,
    target
):

    pred_x = (
        prediction[:, :, :, 1:]
        -
        prediction[:, :, :, :-1]
    )

    pred_y = (
        prediction[:, :, 1:, :]
        -
        prediction[:, :, :-1, :]
    )

    target_x = (
        target[:, :, :, 1:]
        -
        target[:, :, :, :-1]
    )

    target_y = (
        target[:, :, 1:, :]
        -
        target[:, :, :-1, :]
    )

    return (
        F.l1_loss(
            pred_x,
            target_x
        )
        +
        F.l1_loss(
            pred_y,
            target_y
        )
    )


# ============================================================
# TRAINING
# ============================================================

def train():

    print("\n==============================================")
    print("AGBR v2 REAL-DATA TRAINING")
    print("==============================================")

    print(
        f"\nDevice: {DEVICE}"
    )

    if DEVICE == "cpu":

        print(
            "\nWARNING:"
            "\nGPU not detected."
            "\nTraining this model on CPU will be very slow."
            "\nUse Google Colab with GPU for actual training."
        )

    ensure_dir(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # MiDaS
    # --------------------------------------------------------

    print(
        "\nLoading MiDaS..."
    )

    depth_model = DepthEstimator(
        MIDAS_PATH
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = RealPolycamDataset(
        DATASET_DIR,
        TRAIN_SCENES,
        depth_model,
        SAMPLES_PER_FRAME
    )

    val_dataset = RealPolycamDataset(
        DATASET_DIR,
        VAL_SCENES,
        depth_model,
        SAMPLES_PER_FRAME
    )

    print(
        "\nTraining frames:"
        f" {len(train_dataset.frames)}"
    )

    print(
        "Validation frames:"
        f" {len(val_dataset.frames)}"
    )

    print(
        "Test scenes are kept completely separate:"
        f" {TEST_SCENES}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    generator = (
        ResidualBoundaryGenerator()
        .to(DEVICE)
    )

    discriminator = (
        PatchDiscriminator()
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    scheduler_g = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g,
            mode="min",
            factor=0.5,
            patience=4
        )
    )

    scheduler_d = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_d,
            mode="min",
            factor=0.5,
            patience=4
        )
    )

    adversarial_loss = nn.MSELoss()

    best_val = float(
        "inf"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        generator.train()
        discriminator.train()

        total_g = 0.0
        total_d = 0.0

        for (
            inputs,
            targets
        ) in train_loader:

            inputs = inputs.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            coarse = inputs[
                :, 3:4
            ]

            rgb = inputs[
                :, :3
            ]

            # =================================================
            # DISCRIMINATOR
            # =================================================

            optimizer_d.zero_grad()

            with torch.no_grad():

                fake_depth = generator(
                    inputs
                )

            real_pair = torch.cat(
                [
                    rgb,
                    targets
                ],
                dim=1
            )

            fake_pair = torch.cat(
                [
                    rgb,
                    fake_depth
                ],
                dim=1
            )

            real_pred = discriminator(
                real_pair
            )

            fake_pred = discriminator(
                fake_pair
            )

            d_real = adversarial_loss(
                real_pred,
                torch.ones_like(
                    real_pred
                )
            )

            d_fake = adversarial_loss(
                fake_pred,
                torch.zeros_like(
                    fake_pred
                )
            )

            d_loss = 0.5 * (
                d_real
                +
                d_fake
            )

            d_loss.backward()

            optimizer_d.step()

            # =================================================
            # GENERATOR
            # =================================================

            optimizer_g.zero_grad()

            fake_depth = generator(
                inputs
            )

            fake_pair = torch.cat(
                [
                    rgb,
                    fake_depth
                ],
                dim=1
            )

            fake_pred = discriminator(
                fake_pair
            )

            g_adv = adversarial_loss(
                fake_pred,
                torch.ones_like(
                    fake_pred
                )
            )

            g_depth = F.l1_loss(
                fake_depth,
                targets
            )

            g_boundary = (
                boundary_weighted_loss(
                    fake_depth,
                    targets
                )
            )

            g_gradient = (
                gradient_loss(
                    fake_depth,
                    targets
                )
            )

            g_residual = (
                residual_loss(
                    fake_depth,
                    coarse
                )
            )

            g_loss = (
                LAMBDA_ADV
                * g_adv
                +
                LAMBDA_DEPTH
                * g_depth
                +
                LAMBDA_BOUNDARY
                * (
                    g_boundary
                    +
                    0.5
                    * g_gradient
                )
                +
                LAMBDA_RESIDUAL
                * g_residual
            )

            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                generator.parameters(),
                max_norm=1.0
            )

            optimizer_g.step()

            total_g += (
                g_loss.item()
            )

            total_d += (
                d_loss.item()
            )

        # =====================================================
        # VALIDATION
        # =====================================================

        generator.eval()

        val_loss = 0.0
        val_mae = 0.0
        val_count = 0

        with torch.no_grad():

            for (
                inputs,
                targets
            ) in val_loader:

                inputs = inputs.to(
                    DEVICE
                )

                targets = targets.to(
                    DEVICE
                )

                coarse = inputs[
                    :, 3:4
                ]

                prediction = generator(
                    inputs
                )

                loss = F.l1_loss(
                    prediction,
                    targets
                )

                val_loss += (
                    loss.item()
                    *
                    inputs.size(0)
                )

                val_mae += (
                    torch.mean(
                        torch.abs(
                            prediction
                            -
                            targets
                        )
                    ).item()
                    *
                    inputs.size(0)
                )

                val_count += (
                    inputs.size(0)
                )

        val_loss /= max(
            val_count,
            1
        )

        val_mae /= max(
            val_count,
            1
        )

        scheduler_g.step(
            val_loss
        )

        scheduler_d.step(
            total_d
            /
            max(
                len(train_loader),
                1
            )
        )

        print(
            f"\nEpoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        print(
            f"Generator loss: "
            f"{total_g / len(train_loader):.6f}"
        )

        print(
            f"Discriminator loss: "
            f"{total_d / len(train_loader):.6f}"
        )

        print(
            f"Validation L1: "
            f"{val_loss:.6f}"
        )

        print(
            f"Validation MAE: "
            f"{val_mae:.6f}"
        )

        # =====================================================
        # SAVE BEST MODEL
        # =====================================================

        if val_loss < best_val:

            best_val = val_loss

            best_path = os.path.join(
                OUTPUT_DIR,
                "agbr_v2_best.pt"
            )

            torch.save(
                generator.state_dict(),
                best_path
            )

            print(
                f"Saved BEST model:"
                f"\n{best_path}"
            )

        # =====================================================
        # CHECKPOINT
        # =====================================================

        if (
            (epoch + 1) % 10
            == 0
        ):

            checkpoint_path = os.path.join(
                OUTPUT_DIR,
                f"agbr_v2_epoch_{epoch + 1}.pt"
            )

            torch.save(
                generator.state_dict(),
                checkpoint_path
            )

    # =========================================================
    # LOAD BEST MODEL
    # =========================================================

    best_path = os.path.join(
        OUTPUT_DIR,
        "agbr_v2_best.pt"
    )

    generator.load_state_dict(
        torch.load(
            best_path,
            map_location=DEVICE
        )
    )

    generator.eval()

    # =========================================================
    # EXPORT ONNX
    # =========================================================

    onnx_path = os.path.join(
        OUTPUT_DIR,
        "boundary_refinement_gan_v2.onnx"
    )

    dummy_input = torch.randn(
        1,
        5,
        PATCH_SIZE,
        PATCH_SIZE,
        device=DEVICE
    )

    print(
        "\nExporting ONNX..."
    )

    torch.onnx.export(
        generator,
        dummy_input,
        onnx_path,
        input_names=[
            "input"
        ],
        output_names=[
            "output"
        ],
        dynamic_axes={
            "input": {
                0: "batch"
            },
            "output": {
                0: "batch"
            }
        },
        opset_version=13,
        dynamo=False
    )

    print(
        "\n=============================================="
    )

    print(
        "AGBR v2 TRAINING COMPLETE"
    )

    print(
        "=============================================="
    )

    print(
        f"\nBest PyTorch model:"
        f"\n{best_path}"
    )

    print(
        f"\nONNX model:"
        f"\n{onnx_path}"
    )

    print(
        "\nIMPORTANT:"
        "\nThis model expects 5 channels:"
        "\nRGB + coarse depth + confidence"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()