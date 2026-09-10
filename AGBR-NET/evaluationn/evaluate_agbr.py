import os
import sys
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from modules.depth_estimation import DepthEstimator
from modules.confidence_estimator import ConfidenceEstimator
from modules.region_selector import RegionSelector
from modules.gan_refinement import GANRefiner


# =========================================================
# PATHS
# =========================================================

MODEL_MIDAS = os.path.join(
    PROJECT_ROOT,
    "models",
    "midas",
    "midas_v21_small_256.onnx"
)

MODEL_GAN = os.path.join(
    PROJECT_ROOT,
    "models",
    "gan",
    "boundary_refinement_gan.onnx"
)

EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation_data")

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "lidar_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# SAME ALIGNMENT USED IN BASELINE
# =========================================================

HORIZONTAL_FOV_DEG = 69.4
LIDAR_FORWARD_DEG = 180.0
ROW_FRACTION = 0.50


SCENES = [1, 2, 3, 4, 5]


# =========================================================
# ANGLE / PROJECTION
# =========================================================

def wrap_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def lidar_to_image_x(angle_deg, width):

    camera_angle = wrap_angle_deg(
        angle_deg - LIDAR_FORWARD_DEG
    )

    x = (
        width / 2.0
        + (camera_angle / HORIZONTAL_FOV_DEG) * width
    )

    return int(round(x))


# =========================================================
# LOAD DATA
# =========================================================

def load_rgb(scene_no):

    path = os.path.join(
        EVAL_DIR,
        f"scene{scene_no}.jpg"
    )

    frame = cv2.imread(path)

    if frame is None:
        raise FileNotFoundError(path)

    return frame


def load_lidar(scene_no):

    path = os.path.join(
        EVAL_DIR,
        f"scene{scene_no}",
        f"scene_{scene_no:02d}_median_scan.csv"
    )

    return pd.read_csv(path)


# =========================================================
# LIDAR → IMAGE CORRESPONDENCES
# =========================================================

def get_lidar_correspondences(frame, lidar_df):

    H, W = frame.shape[:2]

    xs = []
    depths = []

    for _, row in lidar_df.iterrows():

        angle_deg = float(row["angle_bin_deg"])
        range_m = float(row["median_range_m"])

        camera_angle = wrap_angle_deg(
            angle_deg - LIDAR_FORWARD_DEG
        )

        depth_m = (
            range_m *
            np.cos(np.deg2rad(camera_angle))
        )

        x = lidar_to_image_x(
            angle_deg,
            W
        )

        if x < 0 or x >= W:
            continue

        if not np.isfinite(depth_m):
            continue

        if depth_m <= 0:
            continue

        xs.append(x)
        depths.append(depth_m)

    return np.array(xs), np.array(depths)


# =========================================================
# SAMPLE DEPTH
# =========================================================

def sample_depth(depth, xs):

    H, W = depth.shape

    y_center = int(
        H * ROW_FRACTION
    )

    values = []

    for x in xs:

        x = int(x)

        x0 = max(0, x - 2)
        x1 = min(W, x + 3)

        y0 = max(0, y_center - 5)
        y1 = min(H, y_center + 6)

        patch = depth[
            y0:y1,
            x0:x1
        ]

        valid = patch[
            np.isfinite(patch)
        ]

        if len(valid) == 0:
            values.append(np.nan)
        else:
            values.append(
                float(np.median(valid))
            )

    return np.array(values)


# =========================================================
# POINT ↔ GAN REGION
# =========================================================

def classify_lidar_points(xs, boxes, image_shape):

    H, W = image_shape[:2]

    lidar_y = int(
        H * ROW_FRACTION
    )

    refined = np.zeros(
        len(xs),
        dtype=bool
    )

    for i, x in enumerate(xs):

        for box in boxes:

            x1, y1, x2, y2 = box

            if (
                x1 <= x <= x2
                and y1 <= lidar_y <= y2
            ):
                refined[i] = True
                break

    return refined


# =========================================================
# VISUALIZATION
# =========================================================

def save_region_visualization(
    frame,
    xs,
    refined_mask,
    boxes,
    output_path
):

    vis = frame.copy()

    H, W = vis.shape[:2]

    lidar_y = int(
        H * ROW_FRACTION
    )

    # Draw GAN regions
    for box in boxes:

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            4
        )

    # Draw LiDAR scan line
    cv2.line(
        vis,
        (0, lidar_y),
        (W - 1, lidar_y),
        (255, 0, 0),
        3
    )

    # Draw LiDAR points
    for x, is_refined in zip(
        xs,
        refined_mask
    ):

        x = int(x)

        if is_refined:

            # Refined-region point
            cv2.circle(
                vis,
                (x, lidar_y),
                14,
                (0, 0, 255),
                -1
            )

        else:

            # Unrefined point
            cv2.circle(
                vis,
                (x, lidar_y),
                10,
                (0, 255, 0),
                -1
            )

    cv2.imwrite(
        output_path,
        vis
    )


# =========================================================
# METRICS
# =========================================================

def metrics(prediction, target):

    valid = (
        np.isfinite(prediction)
        &
        np.isfinite(target)
        &
        (target > 0)
    )

    prediction = prediction[valid]
    target = target[valid]

    if len(prediction) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            0
        )

    error = prediction - target

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    absrel = np.mean(
        np.abs(error)
        /
        np.maximum(target, 1e-6)
    )

    return (
        float(mae),
        float(rmse),
        float(absrel),
        len(prediction)
    )


# =========================================================
# COMMON SCALE
# =========================================================

def estimate_midas_scale(
    midas_prediction,
    lidar_depth
):

    valid = (
        np.isfinite(midas_prediction)
        &
        np.isfinite(lidar_depth)
        &
        (midas_prediction > 1e-6)
        &
        (lidar_depth > 0)
    )

    inverse_depth = (
        1.0 /
        midas_prediction[valid]
    )

    lidar_valid = lidar_depth[valid]

    ratios = (
        lidar_valid /
        inverse_depth
    )

    return float(
        np.median(ratios)
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("AGBR-NET + LiDAR REGION-AWARE EVALUATION")
    print("=" * 70)

    print("\nLoading models...")

    depth_model = DepthEstimator(
        MODEL_MIDAS
    )

    confidence_model = ConfidenceEstimator(
        depth_model
    )

    selector = RegionSelector()

    gan = GANRefiner(
        MODEL_GAN
    )

    print("Models loaded successfully.")

    all_results = []

    for scene_no in SCENES:

        print("\n")
        print("=" * 70)
        print(f"PROCESSING SCENE {scene_no}")
        print("=" * 70)

        # -------------------------------------------------
        # RGB
        # -------------------------------------------------

        frame = load_rgb(scene_no)

        H, W = frame.shape[:2]

        # -------------------------------------------------
        # LiDAR
        # -------------------------------------------------

        lidar_df = load_lidar(scene_no)

        xs, lidar_depth = (
            get_lidar_correspondences(
                frame,
                lidar_df
            )
        )

        print(
            f"LiDAR correspondences: {len(xs)}"
        )

        # -------------------------------------------------
        # MiDaS
        # -------------------------------------------------

        print("Running MiDaS...")

        depth = depth_model.predict(
            frame
        )

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        print(
            "Estimating confidence..."
        )

        confidence, _ = (
            confidence_model.estimate(
                frame,
                depth=depth
            )
        )

        # -------------------------------------------------
        # REGION SELECTION
        # -------------------------------------------------

        boxes = selector.select_regions(
            confidence
        )

        coverage = selector.coverage_fraction(
            boxes,
            frame.shape
        )

        print(
            f"Selected GAN regions: {len(boxes)}"
        )

        print(
            f"GAN coverage: "
            f"{coverage * 100:.2f}%"
        )

        # -------------------------------------------------
        # CLASSIFY LiDAR POINTS
        # -------------------------------------------------

        refined_mask = (
            classify_lidar_points(
                xs,
                boxes,
                frame.shape
            )
        )

        refined_count = int(
            np.sum(refined_mask)
        )

        unrefined_count = (
            len(xs)
            - refined_count
        )

        print(
            f"LiDAR points in refined "
            f"regions: "
            f"{refined_count}/{len(xs)}"
        )

        # -------------------------------------------------
        # GAN
        # -------------------------------------------------

        if boxes:

            print(
                "Running Boundary "
                "Refinement GAN..."
            )

            patches = (
                selector.extract_patches(
                    frame,
                    depth,
                    boxes
                )
            )

            refined_patches = (
                gan.refine_all(
                    patches
                )
            )

            final_depth = (
                selector.merge_patches(
                    depth,
                    refined_patches
                )
            )

        else:

            final_depth = depth.copy()

        # -------------------------------------------------
        # SAVE DEPTH
        # -------------------------------------------------

        scene_output = os.path.join(
            OUTPUT_DIR,
            f"scene_{scene_no:02d}"
        )

        os.makedirs(
            scene_output,
            exist_ok=True
        )

        np.save(
            os.path.join(
                scene_output,
                "agbr_refined_depth.npy"
            ),
            final_depth
        )

        # -------------------------------------------------
        # CHECK ACTUAL CHANGE
        # -------------------------------------------------

        difference = np.abs(
            final_depth.astype(
                np.float32
            )
            -
            depth.astype(
                np.float32
            )
        )

        changed = difference > 1e-6

        print(
            f"Changed pixels: "
            f"{np.sum(changed):,}"
        )

        print(
            f"Changed percentage: "
            f"{np.mean(changed) * 100:.4f}%"
        )

        # -------------------------------------------------
        # SAVE VISUALIZATION
        # -------------------------------------------------

        vis_path = os.path.join(
            scene_output,
            "gan_lidar_regions.jpg"
        )

        save_region_visualization(
            frame,
            xs,
            refined_mask,
            boxes,
            vis_path
        )

        print(
            f"Saved visualization:\n"
            f"{vis_path}"
        )

        # -------------------------------------------------
        # SAMPLE DEPTH
        # -------------------------------------------------

        midas_samples = sample_depth(
            depth,
            xs
        )

        agbr_samples = sample_depth(
            final_depth,
            xs
        )

        # -------------------------------------------------
        # COMMON SCALE FROM MiDaS
        # -------------------------------------------------

        scale = (
            estimate_midas_scale(
                midas_samples,
                lidar_depth
            )
        )

        print(
            f"Common scale: {scale:.6f}"
        )

        # Same scale for BOTH methods
        midas_metric = (
            scale /
            np.maximum(
                midas_samples,
                1e-6
            )
        )

        agbr_metric = (
            scale /
            np.maximum(
                agbr_samples,
                1e-6
            )
        )

        # -------------------------------------------------
        # GLOBAL
        # -------------------------------------------------

        g_midas = metrics(
            midas_metric,
            lidar_depth
        )

        g_agbr = metrics(
            agbr_metric,
            lidar_depth
        )

        # -------------------------------------------------
        # REFINED REGION
        # -------------------------------------------------

        if refined_count > 0:

            r_midas = metrics(
                midas_metric[
                    refined_mask
                ],
                lidar_depth[
                    refined_mask
                ]
            )

            r_agbr = metrics(
                agbr_metric[
                    refined_mask
                ],
                lidar_depth[
                    refined_mask
                ]
            )

        else:

            r_midas = (
                np.nan,
                np.nan,
                np.nan,
                0
            )

            r_agbr = r_midas

        # -------------------------------------------------
        # UNREFINED
        # -------------------------------------------------

        if unrefined_count > 0:

            u_midas = metrics(
                midas_metric[
                    ~refined_mask
                ],
                lidar_depth[
                    ~refined_mask
                ]
            )

            u_agbr = metrics(
                agbr_metric[
                    ~refined_mask
                ],
                lidar_depth[
                    ~refined_mask
                ]
            )

        else:

            u_midas = (
                np.nan,
                np.nan,
                np.nan,
                0
            )

            u_agbr = u_midas

        # -------------------------------------------------
        # PRINT
        # -------------------------------------------------

        print("\nGLOBAL")

        print(
            f"MiDaS  MAE={g_midas[0]:.4f} "
            f"RMSE={g_midas[1]:.4f} "
            f"AbsRel={g_midas[2]:.4f}"
        )

        print(
            f"AGBR   MAE={g_agbr[0]:.4f} "
            f"RMSE={g_agbr[1]:.4f} "
            f"AbsRel={g_agbr[2]:.4f}"
        )

        print(
            "\nREFINED LiDAR REGION"
        )

        if refined_count > 0:

            print(
                f"MiDaS  MAE={r_midas[0]:.4f} "
                f"RMSE={r_midas[1]:.4f} "
                f"AbsRel={r_midas[2]:.4f}"
            )

            print(
                f"AGBR   MAE={r_agbr[0]:.4f} "
                f"RMSE={r_agbr[1]:.4f} "
                f"AbsRel={r_agbr[2]:.4f}"
            )

        else:

            print(
                "No LiDAR points "
                "inside refined regions."
            )

        print(
            "\nUNREFINED LiDAR REGION"
        )

        print(
            f"MiDaS  MAE={u_midas[0]:.4f} "
            f"RMSE={u_midas[1]:.4f} "
            f"AbsRel={u_midas[2]:.4f}"
        )

        print(
            f"AGBR   MAE={u_agbr[0]:.4f} "
            f"RMSE={u_agbr[1]:.4f} "
            f"AbsRel={u_agbr[2]:.4f}"
        )

        # -------------------------------------------------
        # SAVE PER-POINT DATA
        # -------------------------------------------------

        point_df = pd.DataFrame({

            "image_x": xs,

            "lidar_depth_m":
                lidar_depth,

            "midas_raw":
                midas_samples,

            "agbr_raw":
                agbr_samples,

            "midas_metric_m":
                midas_metric,

            "agbr_metric_m":
                agbr_metric,

            "refined_region":
                refined_mask

        })

        point_df.to_csv(
            os.path.join(
                scene_output,
                "region_aware_points.csv"
            ),
            index=False
        )

        # -------------------------------------------------
        # STORE SUMMARY
        # -------------------------------------------------

        all_results.append({

            "scene":
                scene_no,

            "N":
                len(xs),

            "GAN_regions":
                len(boxes),

            "GAN_coverage_percent":
                coverage * 100,

            "LiDAR_refined_points":
                refined_count,

            "LiDAR_unrefined_points":
                unrefined_count,

            "common_scale":
                scale,

            "global_MiDaS_MAE":
                g_midas[0],

            "global_AGBR_MAE":
                g_agbr[0],

            "global_MiDaS_RMSE":
                g_midas[1],

            "global_AGBR_RMSE":
                g_agbr[1],

            "global_MiDaS_AbsRel":
                g_midas[2],

            "global_AGBR_AbsRel":
                g_agbr[2],

            "refined_MiDaS_MAE":
                r_midas[0],

            "refined_AGBR_MAE":
                r_agbr[0],

            "refined_MiDaS_RMSE":
                r_midas[1],

            "refined_AGBR_RMSE":
                r_agbr[1],

            "refined_MiDaS_AbsRel":
                r_midas[2],

            "refined_AGBR_AbsRel":
                r_agbr[2],

            "unrefined_MiDaS_MAE":
                u_midas[0],

            "unrefined_AGBR_MAE":
                u_agbr[0],

            "unrefined_MiDaS_RMSE":
                u_midas[1],

            "unrefined_AGBR_RMSE":
                u_agbr[1],

            "unrefined_MiDaS_AbsRel":
                u_midas[2],

            "unrefined_AGBR_AbsRel":
                u_agbr[2]
        })

    # =====================================================
    # FINAL CSV
    # =====================================================

    results_df = pd.DataFrame(
        all_results
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "agbr_region_aware_summary.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FINAL REGION-AWARE AGBR-NET EVALUATION")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(summary_path)

    print("\nDONE.")


if __name__ == "__main__":
    main()