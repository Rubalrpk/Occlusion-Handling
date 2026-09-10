import os
import cv2
import numpy as np
import pandas as pd

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

HFOV_DEG = 69.4
CAMERA_FORWARD_LIDAR_DEG = 180.0


def wrap_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


for scene_no in range(1, 6):

    print("\n" + "=" * 60)
    print(f"Scene {scene_no}")
    print("=" * 60)

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    rgb_path = os.path.join(
        DATASET_DIR,
        f"scene{scene_no}.jpg"
    )

    lidar_path = os.path.join(
        DATASET_DIR,
        f"scene{scene_no}",
        f"scene_{scene_no:02d}_median_scan.csv"
    )

    output_dir = os.path.join(
        OUTPUT_DIR,
        f"scene_{scene_no:02d}"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    rgb = cv2.imread(rgb_path)

    lidar = pd.read_csv(
        lidar_path
    )

    if rgb is None:
        print("RGB missing — skipping.")
        continue

    height, width = rgb.shape[:2]

    # --------------------------------------------------------
    # LIDAR → CAMERA ANGLE
    # --------------------------------------------------------

    lidar["camera_angle_deg"] = (
        lidar["angle_bin_deg"]
        .apply(
            lambda a:
            wrap_angle(
                a -
                CAMERA_FORWARD_LIDAR_DEG
            )
        )
    )

    # --------------------------------------------------------
    # KEEP CAMERA FOV
    # --------------------------------------------------------

    visible = lidar[
        np.abs(
            lidar["camera_angle_deg"]
        )
        <= HFOV_DEG / 2
    ].copy()

    # --------------------------------------------------------
    # CAMERA ANGLE → IMAGE X
    # --------------------------------------------------------

    focal_px = (
        width / 2
    ) / np.tan(
        np.radians(
            HFOV_DEG / 2
        )
    )

    visible["pixel_x"] = (
        width / 2
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
    # SCAN LINE
    # --------------------------------------------------------

    scan_y = height // 2

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    preview = rgb.copy()

    # Green = assumed LiDAR scan plane

    cv2.line(
        preview,
        (0, scan_y),
        (width - 1, scan_y),
        (0, 255, 0),
        3
    )

    # Red = projected LiDAR measurements

    for x in visible["pixel_x"]:

        cv2.circle(
            preview,
            (int(x), scan_y),
            7,
            (0, 0, 255),
            -1
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        "rgb_lidar_projection.jpg"
    )

    cv2.imwrite(
        output_path,
        preview
    )

    print(
        "Saved:"
    )

    print(
        output_path
    )

print("\nAll projection images generated.")