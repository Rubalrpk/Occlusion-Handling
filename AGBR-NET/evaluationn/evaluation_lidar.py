import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    ROOT,
    "evaluation_data"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs",
    "lidar_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CAMERA / LiDAR PARAMETERS
# ============================================================

HFOV_DEG = 69.4

CAMERA_FORWARD_LIDAR_DEG = 180.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def load_scene(scene_no):

    rgb_path = os.path.join(
        DATASET_DIR,
        f"scene{scene_no}.jpg"
    )

    lidar_path = os.path.join(
        DATASET_DIR,
        f"scene{scene_no}",
        f"scene_{scene_no:02d}_median_scan.csv"
    )

    depth_path = os.path.join(
        OUTPUT_DIR,
        f"scene_{scene_no:02d}",
        "midas_raw.npy"
    )

    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(
            f"RGB missing:\n{rgb_path}"
        )

    if not os.path.isfile(lidar_path):
        raise FileNotFoundError(
            f"LiDAR missing:\n{lidar_path}"
        )

    if not os.path.isfile(depth_path):
        raise FileNotFoundError(
            f"MiDaS depth missing:\n{depth_path}"
        )

    rgb = cv2.imread(rgb_path)

    depth = np.load(depth_path)

    lidar = pd.read_csv(lidar_path)

    return rgb, depth, lidar


# ============================================================
# CREATE RGB-LiDAR CORRESPONDENCES
# ============================================================

def create_correspondence(rgb, depth, lidar):

    height, width = rgb.shape[:2]

    lidar = lidar.copy()

    # --------------------------------------------------------
    # LiDAR angle -> camera angle
    # --------------------------------------------------------

    lidar["camera_angle_deg"] = (
        lidar["angle_bin_deg"]
        .apply(
            lambda angle:
            wrap_angle(
                angle -
                CAMERA_FORWARD_LIDAR_DEG
            )
        )
    )

    # --------------------------------------------------------
    # Keep points inside camera FOV
    # --------------------------------------------------------

    visible = lidar[
        np.abs(
            lidar["camera_angle_deg"]
        ) <= HFOV_DEG / 2.0
    ].copy()

    # --------------------------------------------------------
    # Camera angle -> image X
    # --------------------------------------------------------

    focal_px = (
        width / 2.0
    ) / np.tan(
        np.radians(
            HFOV_DEG / 2.0
        )
    )

    visible["pixel_x"] = (
        width / 2.0
        +
        focal_px
        *
        np.tan(
            np.radians(
                visible["camera_angle_deg"]
            )
        )
    )

    visible["pixel_x"] = (
        visible["pixel_x"]
        .round()
        .astype(int)
        .clip(
            0,
            width - 1
        )
    )

    # --------------------------------------------------------
    # LiDAR scan row
    # --------------------------------------------------------

    scan_y = height // 2

    # --------------------------------------------------------
    # Sample MiDaS
    # --------------------------------------------------------

    midas_values = []

    for x in visible["pixel_x"]:

        y1 = max(
            0,
            scan_y - 5
        )

        y2 = min(
            height,
            scan_y + 6
        )

        patch = depth[
            y1:y2,
            x
        ]

        patch = patch[
            np.isfinite(patch)
        ]

        if len(patch) == 0:

            midas_values.append(
                np.nan
            )

        else:

            midas_values.append(
                float(
                    np.median(patch)
                )
            )

    visible["midas_relative"] = (
        midas_values
    )

    # --------------------------------------------------------
    # Convert radial LiDAR range to camera-axis depth
    # --------------------------------------------------------

    visible["lidar_depth_m"] = (
        visible["median_range_m"]
        *
        np.cos(
            np.radians(
                visible["camera_angle_deg"]
            )
        )
    )

    # --------------------------------------------------------
    # Valid measurements
    # --------------------------------------------------------

    comparison = visible[
        np.isfinite(
            visible["midas_relative"]
        )
        &
        np.isfinite(
            visible["lidar_depth_m"]
        )
        &
        (
            visible["lidar_depth_m"] > 0
        )
        &
        (
            visible["midas_relative"] > 0
        )
    ].copy()

    return comparison


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(gt, pred):

    valid = (
        np.isfinite(gt)
        &
        np.isfinite(pred)
        &
        (gt > 0)
        &
        (pred > 0)
    )

    gt = gt[valid]
    pred = pred[valid]

    if len(gt) == 0:

        return {
            "MAE_m": np.nan,
            "RMSE_m": np.nan,
            "AbsRel": np.nan,
            "N": 0
        }

    error = pred - gt

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    abs_rel = np.mean(
        np.abs(error) / gt
    )

    return {
        "MAE_m": mae,
        "RMSE_m": rmse,
        "AbsRel": abs_rel,
        "N": len(gt)
    }


# ============================================================
# LOAD ALL SCENES
# ============================================================

scene_data = {}

print("\n" + "=" * 70)
print("MiDaS + LiDAR SCALE-ALIGNED BASELINE")
print("=" * 70)

for scene_no in range(1, 6):

    print(
        f"\nLoading Scene {scene_no}..."
    )

    rgb, depth, lidar = load_scene(
        scene_no
    )

    comparison = create_correspondence(
        rgb,
        depth,
        lidar
    )

    scene_data[scene_no] = {
        "rgb": rgb,
        "depth": depth,
        "comparison": comparison
    }

    print(
        f"Valid correspondences: "
        f"{len(comparison)}"
    )


# ============================================================
# PER-SCENE SCALE + INVERSION ALIGNMENT
# ============================================================

results = []


for scene_no in range(1, 6):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING SCENE {scene_no}"
    )
    print("=" * 70)

    comparison = scene_data[
        scene_no
    ]["comparison"].copy()

    # --------------------------------------------------------
    # MiDaS is treated as relative inverse-depth-like output.
    #
    # Convert it to a depth-like quantity:
    #
    # inverse = 1 / MiDaS
    # --------------------------------------------------------

    comparison["midas_inverse"] = (
        1.0 /
        comparison["midas_relative"]
    )

    # --------------------------------------------------------
    # Robust scale alignment
    #
    # metric depth = scale * inverse(MiDaS)
    #
    # Use median ratio so that a few noisy LiDAR points
    # do not dominate the scale.
    # --------------------------------------------------------

    lidar_values = comparison[
        "lidar_depth_m"
    ].to_numpy()

    inverse_values = comparison[
        "midas_inverse"
    ].to_numpy()

    valid = (
        np.isfinite(
            lidar_values
        )
        &
        np.isfinite(
            inverse_values
        )
        &
        (inverse_values > 0)
    )

    lidar_values = lidar_values[
        valid
    ]

    inverse_values = inverse_values[
        valid
    ]

    if len(lidar_values) < 3:

        print(
            "[SKIP] Not enough valid points."
        )

        continue

    scale = np.median(
        lidar_values /
        inverse_values
    )

    # --------------------------------------------------------
    # Generate aligned metric prediction
    # --------------------------------------------------------

    comparison[
        "midas_metric_aligned_m"
    ] = (
        scale *
        comparison[
            "midas_inverse"
        ]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    gt = comparison[
        "lidar_depth_m"
    ].to_numpy()

    pred = comparison[
        "midas_metric_aligned_m"
    ].to_numpy()

    metrics = calculate_metrics(
        gt,
        pred
    )

    print(
        f"Scale: {scale:.8f}"
    )

    print(
        f"MAE   : "
        f"{metrics['MAE_m']:.4f} m"
    )

    print(
        f"RMSE  : "
        f"{metrics['RMSE_m']:.4f} m"
    )

    print(
        f"AbsRel: "
        f"{metrics['AbsRel']:.4f}"
    )

    print(
        f"N     : "
        f"{metrics['N']}"
    )

    # --------------------------------------------------------
    # Save comparison CSV
    # --------------------------------------------------------

    scene_output_dir = os.path.join(
        OUTPUT_DIR,
        f"scene_{scene_no:02d}"
    )

    os.makedirs(
        scene_output_dir,
        exist_ok=True
    )

    csv_path = os.path.join(
        scene_output_dir,
        "midas_scale_comparison.csv"
    )

    comparison.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        scene_output_dir,
        "midas_scale_metrics.txt"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        f.write(
            f"Scene: {scene_no}\n"
        )

        f.write(
            "Alignment: "
            "metric_depth = "
            f"{scale:.10f} / MiDaS\n"
        )

        f.write(
            f"MAE_m: "
            f"{metrics['MAE_m']:.8f}\n"
        )

        f.write(
            f"RMSE_m: "
            f"{metrics['RMSE_m']:.8f}\n"
        )

        f.write(
            f"AbsRel: "
            f"{metrics['AbsRel']:.8f}\n"
        )

        f.write(
            f"N: "
            f"{metrics['N']}\n"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        comparison["pixel_x"],
        comparison["lidar_depth_m"],
        marker="o",
        label="LiDAR reference"
    )

    plt.plot(
        comparison["pixel_x"],
        comparison[
            "midas_metric_aligned_m"
        ],
        marker="x",
        label="MiDaS aligned"
    )

    plt.xlabel(
        "RGB horizontal pixel"
    )

    plt.ylabel(
        "Depth (m)"
    )

    plt.title(
        f"Scene {scene_no} — "
        "MiDaS vs LiDAR"
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(
        scene_output_dir,
        "scale_lidar_vs_midas.png"
    )

    plt.savefig(
        plot_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    results.append(
        {
            "scene": scene_no,
            "scale": scale,
            "MAE_m": metrics[
                "MAE_m"
            ],
            "RMSE_m": metrics[
                "RMSE_m"
            ],
            "AbsRel": metrics[
                "AbsRel"
            ],
            "N": metrics[
                "N"
            ]
        }
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    results
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "midas_scale_baseline_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 70)
print(
    "FINAL MiDaS SCALE-ALIGNED BASELINE"
)
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nSaved:"
)

print(
    summary_path
)

print(
    "\nDONE."
)