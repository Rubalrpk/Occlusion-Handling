import os
import cv2
import numpy as np
import pandas as pd

from modules.depth_estimation import DepthEstimator
from modules.confidence_estimator import ConfidenceEstimator
from modules.region_selector import RegionSelector
from modules.gan_refinement import GANRefiner


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = r"C:\Users\rubal\OneDrive\Desktop\urf"

PROJECT_DIR = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET"

MIDAS_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "midas",
    "midas_v21_small_256.onnx"
)

GAN_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "gan",
    "boundary_refinement_gan.onnx"
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

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "all_scenes_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, ground_truth, mask):

    valid = (
        mask
        & np.isfinite(prediction)
        & np.isfinite(ground_truth)
        & (ground_truth > 0)
    )

    p = prediction[valid].astype(np.float64)
    g = ground_truth[valid].astype(np.float64)

    if len(g) == 0:
        return np.nan, np.nan, np.nan, 0

    error = p - g

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    abs_rel = np.mean(
        np.abs(error)
        / np.maximum(g, 1e-6)
    )

    return mae, rmse, abs_rel, len(g)


# ============================================================
# CONTROLLED SCALE ALIGNMENT
# ============================================================

def align_prediction(prediction, ground_truth, mask):

    valid = (
        mask
        & np.isfinite(prediction)
        & np.isfinite(ground_truth)
        & (ground_truth > 0)
    )

    p = prediction[valid].astype(np.float64)
    g = ground_truth[valid].astype(np.float64)

    if len(g) == 0:
        raise RuntimeError(
            "No valid pixels available for alignment."
        )

    # --------------------------------------------------------
    # DIRECT ORIENTATION
    # --------------------------------------------------------

    denominator_direct = np.sum(p * p)

    if denominator_direct > 1e-12:

        scale_direct = (
            np.sum(p * g)
            / denominator_direct
        )

        error_direct = np.mean(
            np.abs(
                scale_direct * p - g
            )
        )

    else:

        scale_direct = 0.0
        error_direct = np.inf

    # --------------------------------------------------------
    # INVERTED ORIENTATION
    # --------------------------------------------------------

    prediction_inverted = (
        prediction.max()
        - prediction
    )

    p_inv = prediction_inverted[
        valid
    ].astype(np.float64)

    denominator_inverse = np.sum(
        p_inv * p_inv
    )

    if denominator_inverse > 1e-12:

        scale_inverse = (
            np.sum(p_inv * g)
            / denominator_inverse
        )

        error_inverse = np.mean(
            np.abs(
                scale_inverse * p_inv - g
            )
        )

    else:

        scale_inverse = 0.0
        error_inverse = np.inf

    # --------------------------------------------------------
    # SELECT ORIENTATION
    # --------------------------------------------------------

    if error_direct <= error_inverse:

        aligned = (
            scale_direct
            * prediction
        )

        orientation = "direct"
        scale = scale_direct

    else:

        aligned = (
            scale_inverse
            * (
                prediction.max()
                - prediction
            )
        )

        orientation = "inverted"
        scale = scale_inverse

    return (
        aligned.astype(np.float32),
        orientation,
        scale
    )


# ============================================================
# IMPROVEMENT
# ============================================================

def improvement(before, after):

    if (
        np.isnan(before)
        or before == 0
    ):
        return np.nan

    return (
        (before - after)
        / before
        * 100.0
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

print("\n==============================================")
print("AGBR ALL-SCENE LiDAR EVALUATION")
print("==============================================")

for path in [
    MIDAS_PATH,
    GAN_PATH
]:

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"\nRequired model not found:\n{path}"
        )


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print("\nLoading models...")

depth_model = DepthEstimator(
    MIDAS_PATH
)

confidence_model = ConfidenceEstimator(
    depth_model
)

selector = RegionSelector()

gan = GANRefiner(
    GAN_PATH
)

print("Models loaded.")


# ============================================================
# STORAGE
# ============================================================

frame_results = []


# ============================================================
# PROCESS ALL SCENES
# ============================================================

for scene_name in SCENES:

    print("\n")
    print("==============================================")
    print(f"PROCESSING {scene_name.upper()}")
    print("==============================================")

    scene_dir = os.path.join(
        DATASET_DIR,
        scene_name
    )

    rgb_dir = os.path.join(
        scene_dir,
        "keyframes",
        "images"
    )

    depth_dir = os.path.join(
        scene_dir,
        "keyframes",
        "depth"
    )

    # --------------------------------------------------------
    # CHECK FOLDERS
    # --------------------------------------------------------

    if not os.path.isdir(rgb_dir):

        print(
            f"RGB folder missing:\n{rgb_dir}"
        )

        continue

    if not os.path.isdir(depth_dir):

        print(
            f"Depth folder missing:\n{depth_dir}"
        )

        continue

    # --------------------------------------------------------
    # FIND MATCHED RGB + DEPTH FRAMES
    # --------------------------------------------------------

    rgb_files = {
        os.path.splitext(filename)[0]: filename
        for filename in os.listdir(rgb_dir)
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    }

    depth_files = {
        os.path.splitext(filename)[0]: filename
        for filename in os.listdir(depth_dir)
        if filename.lower().endswith(".png")
    }

    frame_ids = sorted(
        set(rgb_files.keys())
        &
        set(depth_files.keys())
    )

    print(
        f"Matched frames: {len(frame_ids)}"
    )

    if len(frame_ids) == 0:
        continue

    # --------------------------------------------------------
    # PROCESS EACH FRAME
    # --------------------------------------------------------

    for frame_number, frame_id in enumerate(
        frame_ids,
        start=1
    ):

        print(
            f"\n[{scene_name}] "
            f"Frame {frame_number}/{len(frame_ids)} "
            f"ID={frame_id}"
        )

        try:

            # =================================================
            # LOAD RGB
            # =================================================

            rgb_path = os.path.join(
                rgb_dir,
                rgb_files[frame_id]
            )

            frame = cv2.imread(
                rgb_path
            )

            if frame is None:

                print(
                    "  ERROR: RGB could not be read."
                )

                continue

            # =================================================
            # LOAD POLYCAM LiDAR DEPTH
            # =================================================

            depth_path = os.path.join(
                depth_dir,
                depth_files[frame_id]
            )

            lidar_raw = cv2.imread(
                depth_path,
                cv2.IMREAD_UNCHANGED
            )

            if lidar_raw is None:

                print(
                    "  ERROR: LiDAR depth could not be read."
                )

                continue

            # Polycam depth is stored in millimetres
            lidar_depth = (
                lidar_raw.astype(
                    np.float32
                )
                / 1000.0
            )

            # =================================================
            # MiDaS
            # =================================================

            midas_depth = (
                depth_model.predict(
                    frame
                )
            )

            # =================================================
            # CONFIDENCE
            # =================================================

            confidence, _ = (
                confidence_model.estimate(
                    frame,
                    depth=midas_depth
                )
            )

            # =================================================
            # ADAPTIVE REGION SELECTION
            # =================================================

            boxes = (
                selector.select_regions(
                    confidence
                )
            )

            coverage = (
                selector.coverage_fraction(
                    boxes,
                    frame.shape
                )
            )

            # =================================================
            # CREATE REFINED REGION MASK
            # =================================================

            H, W = frame.shape[:2]

            refined_mask_full = np.zeros(
                (H, W),
                dtype=np.uint8
            )

            for (
                x0,
                y0,
                x1,
                y1
            ) in boxes:

                refined_mask_full[
                    y0:y1,
                    x0:x1
                ] = 1

            # =================================================
            # GAN REFINEMENT
            # =================================================

            patches = (
                selector.extract_patches(
                    frame,
                    midas_depth,
                    boxes
                )
            )

            refined_patches = (
                gan.refine_all(
                    patches
                )
            )

            agbr_depth = (
                selector.merge_patches(
                    midas_depth,
                    refined_patches
                )
            )

            # =================================================
            # RESIZE DEPTH PREDICTIONS TO LiDAR RESOLUTION
            # =================================================

            lidar_h, lidar_w = (
                lidar_depth.shape
            )

            midas_eval = cv2.resize(
                midas_depth,
                (lidar_w, lidar_h),
                interpolation=cv2.INTER_AREA
            )

            agbr_eval = cv2.resize(
                agbr_depth,
                (lidar_w, lidar_h),
                interpolation=cv2.INTER_AREA
            )

            refined_mask = cv2.resize(
                refined_mask_full,
                (lidar_w, lidar_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # =================================================
            # VALID LiDAR MASK
            # =================================================

            lidar_valid = (
                np.isfinite(lidar_depth)
                &
                (lidar_depth > 0)
            )

            global_mask = lidar_valid

            refined_eval_mask = (
                lidar_valid
                &
                refined_mask
            )

            unrefined_eval_mask = (
                lidar_valid
                &
                (~refined_mask)
            )

            # =================================================
            # FIT ALIGNMENT USING MiDaS ONLY
            # =================================================

            (
                midas_aligned,
                orientation,
                scale
            ) = align_prediction(
                midas_eval,
                lidar_depth,
                global_mask
            )

            # =================================================
            # APPLY EXACT SAME TRANSFORMATION TO AGBR
            # =================================================

            if orientation == "direct":

                agbr_aligned = (
                    scale
                    * agbr_eval
                )

            else:

                agbr_aligned = (
                    scale
                    * (
                        midas_eval.max()
                        - agbr_eval
                    )
                )

            agbr_aligned = (
                agbr_aligned.astype(
                    np.float32
                )
            )

            # =================================================
            # GLOBAL METRICS
            # =================================================

            midas_global = calculate_metrics(
                midas_aligned,
                lidar_depth,
                global_mask
            )

            agbr_global = calculate_metrics(
                agbr_aligned,
                lidar_depth,
                global_mask
            )

            # =================================================
            # REFINED REGION METRICS
            # =================================================

            midas_refined = calculate_metrics(
                midas_aligned,
                lidar_depth,
                refined_eval_mask
            )

            agbr_refined = calculate_metrics(
                agbr_aligned,
                lidar_depth,
                refined_eval_mask
            )

            # =================================================
            # UNREFINED REGION METRICS
            # =================================================

            midas_unrefined = calculate_metrics(
                midas_aligned,
                lidar_depth,
                unrefined_eval_mask
            )

            agbr_unrefined = calculate_metrics(
                agbr_aligned,
                lidar_depth,
                unrefined_eval_mask
            )

            # =================================================
            # GAN DEPTH CHANGE
            # =================================================

            depth_change = np.abs(
                agbr_depth.astype(
                    np.float32
                )
                -
                midas_depth.astype(
                    np.float32
                )
            )

            changed_pixels = np.sum(
                depth_change > 1e-6
            )

            changed_percentage = (
                changed_pixels
                /
                depth_change.size
                * 100.0
            )

            # =================================================
            # STORE RESULT
            # =================================================

            frame_results.append({

                "scene":
                    scene_name,

                "frame_id":
                    frame_id,

                "num_regions":
                    len(boxes),

                "gan_coverage_percent":
                    coverage * 100.0,

                "gan_changed_percent":
                    changed_percentage,

                "orientation":
                    orientation,

                "scale":
                    scale,

                # ---------------------------------------------
                # GLOBAL
                # ---------------------------------------------

                "midas_global_mae":
                    midas_global[0],

                "agbr_global_mae":
                    agbr_global[0],

                "global_mae_improvement_percent":
                    improvement(
                        midas_global[0],
                        agbr_global[0]
                    ),

                "midas_global_rmse":
                    midas_global[1],

                "agbr_global_rmse":
                    agbr_global[1],

                "global_rmse_improvement_percent":
                    improvement(
                        midas_global[1],
                        agbr_global[1]
                    ),

                "midas_global_absrel":
                    midas_global[2],

                "agbr_global_absrel":
                    agbr_global[2],

                "global_absrel_improvement_percent":
                    improvement(
                        midas_global[2],
                        agbr_global[2]
                    ),

                # ---------------------------------------------
                # REFINED
                # ---------------------------------------------

                "midas_refined_mae":
                    midas_refined[0],

                "agbr_refined_mae":
                    agbr_refined[0],

                "refined_mae_improvement_percent":
                    improvement(
                        midas_refined[0],
                        agbr_refined[0]
                    ),

                "midas_refined_rmse":
                    midas_refined[1],

                "agbr_refined_rmse":
                    agbr_refined[1],

                "refined_rmse_improvement_percent":
                    improvement(
                        midas_refined[1],
                        agbr_refined[1]
                    ),

                "midas_refined_absrel":
                    midas_refined[2],

                "agbr_refined_absrel":
                    agbr_refined[2],

                "refined_absrel_improvement_percent":
                    improvement(
                        midas_refined[2],
                        agbr_refined[2]
                    ),

                # ---------------------------------------------
                # UNREFINED
                # ---------------------------------------------

                "midas_unrefined_mae":
                    midas_unrefined[0],

                "agbr_unrefined_mae":
                    agbr_unrefined[0],

                "unrefined_mae_improvement_percent":
                    improvement(
                        midas_unrefined[0],
                        agbr_unrefined[0]
                    ),

                "midas_unrefined_rmse":
                    midas_unrefined[1],

                "agbr_unrefined_rmse":
                    agbr_unrefined[1],

                "unrefined_rmse_improvement_percent":
                    improvement(
                        midas_unrefined[1],
                        agbr_unrefined[1]
                    ),

                "midas_unrefined_absrel":
                    midas_unrefined[2],

                "agbr_unrefined_absrel":
                    agbr_unrefined[2],

                "unrefined_absrel_improvement_percent":
                    improvement(
                        midas_unrefined[2],
                        agbr_unrefined[2]
                    ),
            })

            # =================================================
            # PRINT FRAME RESULT
            # =================================================

            print(
                f"  Regions: {len(boxes)} | "
                f"Coverage: {coverage * 100:.2f}% | "
                f"Global MAE: "
                f"{midas_global[0]:.4f} -> "
                f"{agbr_global[0]:.4f} | "
                f"Refined MAE: "
                f"{midas_refined[0]:.4f} -> "
                f"{agbr_refined[0]:.4f}"
            )

        except Exception as e:

            print(
                f"  ERROR in frame {frame_id}: {e}"
            )

            continue


# ============================================================
# CHECK RESULTS
# ============================================================

print("\n\n==============================================")
print("CREATING RESULTS")
print("==============================================")

if len(frame_results) == 0:

    raise RuntimeError(
        "No frames were successfully evaluated."
    )

df = pd.DataFrame(
    frame_results
)


# ============================================================
# SAVE FRAME-LEVEL RESULTS
# ============================================================

frame_csv = os.path.join(
    OUTPUT_DIR,
    "frame_level_results.csv"
)

df.to_csv(
    frame_csv,
    index=False
)


# ============================================================
# SCENE SUMMARY
# ============================================================

scene_summary = []

for scene_name in SCENES:

    scene_data = df[
        df["scene"] == scene_name
    ]

    if len(scene_data) == 0:
        continue

    scene_summary.append({

        "scene":
            scene_name,

        "frames":
            len(scene_data),

        "avg_regions":
            scene_data[
                "num_regions"
            ].mean(),

        "avg_gan_coverage_percent":
            scene_data[
                "gan_coverage_percent"
            ].mean(),

        "avg_gan_changed_percent":
            scene_data[
                "gan_changed_percent"
            ].mean(),

        # GLOBAL
        "midas_global_mae":
            scene_data[
                "midas_global_mae"
            ].mean(),

        "agbr_global_mae":
            scene_data[
                "agbr_global_mae"
            ].mean(),

        "global_mae_improvement_percent":
            scene_data[
                "global_mae_improvement_percent"
            ].mean(),

        "midas_global_rmse":
            scene_data[
                "midas_global_rmse"
            ].mean(),

        "agbr_global_rmse":
            scene_data[
                "agbr_global_rmse"
            ].mean(),

        "global_rmse_improvement_percent":
            scene_data[
                "global_rmse_improvement_percent"
            ].mean(),

        "midas_global_absrel":
            scene_data[
                "midas_global_absrel"
            ].mean(),

        "agbr_global_absrel":
            scene_data[
                "agbr_global_absrel"
            ].mean(),

        "global_absrel_improvement_percent":
            scene_data[
                "global_absrel_improvement_percent"
            ].mean(),

        # REFINED
        "midas_refined_mae":
            scene_data[
                "midas_refined_mae"
            ].mean(),

        "agbr_refined_mae":
            scene_data[
                "agbr_refined_mae"
            ].mean(),

        "refined_mae_improvement_percent":
            scene_data[
                "refined_mae_improvement_percent"
            ].mean(),

        "midas_refined_rmse":
            scene_data[
                "midas_refined_rmse"
            ].mean(),

        "agbr_refined_rmse":
            scene_data[
                "agbr_refined_rmse"
            ].mean(),

        "refined_rmse_improvement_percent":
            scene_data[
                "refined_rmse_improvement_percent"
            ].mean(),

        "midas_refined_absrel":
            scene_data[
                "midas_refined_absrel"
            ].mean(),

        "agbr_refined_absrel":
            scene_data[
                "agbr_refined_absrel"
            ].mean(),

        "refined_absrel_improvement_percent":
            scene_data[
                "refined_absrel_improvement_percent"
            ].mean(),

        # UNREFINED
        "midas_unrefined_mae":
            scene_data[
                "midas_unrefined_mae"
            ].mean(),

        "agbr_unrefined_mae":
            scene_data[
                "agbr_unrefined_mae"
            ].mean(),

        "unrefined_mae_improvement_percent":
            scene_data[
                "unrefined_mae_improvement_percent"
            ].mean(),

        "midas_unrefined_rmse":
            scene_data[
                "midas_unrefined_rmse"
            ].mean(),

        "agbr_unrefined_rmse":
            scene_data[
                "agbr_unrefined_rmse"
            ].mean(),

        "unrefined_rmse_improvement_percent":
            scene_data[
                "unrefined_rmse_improvement_percent"
            ].mean(),

        "midas_unrefined_absrel":
            scene_data[
                "midas_unrefined_absrel"
            ].mean(),

        "agbr_unrefined_absrel":
            scene_data[
                "agbr_unrefined_absrel"
            ].mean(),

        "unrefined_absrel_improvement_percent":
            scene_data[
                "unrefined_absrel_improvement_percent"
            ].mean(),
    })


scene_df = pd.DataFrame(
    scene_summary
)


# ============================================================
# SAVE SCENE SUMMARY
# ============================================================

scene_csv = os.path.join(
    OUTPUT_DIR,
    "scene_summary.csv"
)

scene_df.to_csv(
    scene_csv,
    index=False
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall = {

    "total_frames":
        len(df),

    "scenes_evaluated":
        df["scene"].nunique(),

    "average_gan_coverage_percent":
        df[
            "gan_coverage_percent"
        ].mean(),

    "average_gan_changed_percent":
        df[
            "gan_changed_percent"
        ].mean(),

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    "midas_global_mae":
        df[
            "midas_global_mae"
        ].mean(),

    "agbr_global_mae":
        df[
            "agbr_global_mae"
        ].mean(),

    "global_mae_improvement_percent":
        df[
            "global_mae_improvement_percent"
        ].mean(),

    "midas_global_rmse":
        df[
            "midas_global_rmse"
        ].mean(),

    "agbr_global_rmse":
        df[
            "agbr_global_rmse"
        ].mean(),

    "global_rmse_improvement_percent":
        df[
            "global_rmse_improvement_percent"
        ].mean(),

    "midas_global_absrel":
        df[
            "midas_global_absrel"
        ].mean(),

    "agbr_global_absrel":
        df[
            "agbr_global_absrel"
        ].mean(),

    "global_absrel_improvement_percent":
        df[
            "global_absrel_improvement_percent"
        ].mean(),

    # --------------------------------------------------------
    # REFINED
    # --------------------------------------------------------

    "midas_refined_mae":
        df[
            "midas_refined_mae"
        ].mean(),

    "agbr_refined_mae":
        df[
            "agbr_refined_mae"
        ].mean(),

    "refined_mae_improvement_percent":
        df[
            "refined_mae_improvement_percent"
        ].mean(),

    "midas_refined_rmse":
        df[
            "midas_refined_rmse"
        ].mean(),

    "agbr_refined_rmse":
        df[
            "agbr_refined_rmse"
        ].mean(),

    "refined_rmse_improvement_percent":
        df[
            "refined_rmse_improvement_percent"
        ].mean(),

    "midas_refined_absrel":
        df[
            "midas_refined_absrel"
        ].mean(),

    "agbr_refined_absrel":
        df[
            "agbr_refined_absrel"
        ].mean(),

    "refined_absrel_improvement_percent":
        df[
            "refined_absrel_improvement_percent"
        ].mean(),

    # --------------------------------------------------------
    # UNREFINED
    # --------------------------------------------------------

    "midas_unrefined_mae":
        df[
            "midas_unrefined_mae"
        ].mean(),

    "agbr_unrefined_mae":
        df[
            "agbr_unrefined_mae"
        ].mean(),

    "unrefined_mae_improvement_percent":
        df[
            "unrefined_mae_improvement_percent"
        ].mean(),

    "midas_unrefined_rmse":
        df[
            "midas_unrefined_rmse"
        ].mean(),

    "agbr_unrefined_rmse":
        df[
            "agbr_unrefined_rmse"
        ].mean(),

    "unrefined_rmse_improvement_percent":
        df[
            "unrefined_rmse_improvement_percent"
        ].mean(),

    "midas_unrefined_absrel":
        df[
            "midas_unrefined_absrel"
        ].mean(),

    "agbr_unrefined_absrel":
        df[
            "agbr_unrefined_absrel"
        ].mean(),

    "unrefined_absrel_improvement_percent":
        df[
            "unrefined_absrel_improvement_percent"
        ].mean(),
}


# ============================================================
# SAVE OVERALL SUMMARY
# ============================================================

overall_csv = os.path.join(
    OUTPUT_DIR,
    "overall_summary.csv"
)

pd.DataFrame(
    [overall]
).to_csv(
    overall_csv,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("FINAL OVERALL RESULTS")
print("==============================================")

print(
    f"Total frames evaluated: "
    f"{overall['total_frames']}"
)

print(
    f"Scenes evaluated: "
    f"{overall['scenes_evaluated']}"
)

print(
    f"\nAverage GAN coverage: "
    f"{overall['average_gan_coverage_percent']:.2f}%"
)

print(
    f"Average changed pixels: "
    f"{overall['average_gan_changed_percent']:.2f}%"
)


print("\nGLOBAL")

print(
    f"MAE: "
    f"{overall['midas_global_mae']:.6f}"
    f" -> "
    f"{overall['agbr_global_mae']:.6f} m "
    f"({overall['global_mae_improvement_percent']:+.2f}%)"
)

print(
    f"RMSE: "
    f"{overall['midas_global_rmse']:.6f}"
    f" -> "
    f"{overall['agbr_global_rmse']:.6f} m "
    f"({overall['global_rmse_improvement_percent']:+.2f}%)"
)

print(
    f"AbsRel: "
    f"{overall['midas_global_absrel']:.6f}"
    f" -> "
    f"{overall['agbr_global_absrel']:.6f} "
    f"({overall['global_absrel_improvement_percent']:+.2f}%)"
)


print("\nREFINED GAN REGIONS")

print(
    f"MAE: "
    f"{overall['midas_refined_mae']:.6f}"
    f" -> "
    f"{overall['agbr_refined_mae']:.6f} m "
    f"({overall['refined_mae_improvement_percent']:+.2f}%)"
)

print(
    f"RMSE: "
    f"{overall['midas_refined_rmse']:.6f}"
    f" -> "
    f"{overall['agbr_refined_rmse']:.6f} m "
    f"({overall['refined_rmse_improvement_percent']:+.2f}%)"
)

print(
    f"AbsRel: "
    f"{overall['midas_refined_absrel']:.6f}"
    f" -> "
    f"{overall['agbr_refined_absrel']:.6f} "
    f"({overall['refined_absrel_improvement_percent']:+.2f}%)"
)


print("\nUNREFINED REGIONS")

print(
    f"MAE: "
    f"{overall['midas_unrefined_mae']:.6f}"
    f" -> "
    f"{overall['agbr_unrefined_mae']:.6f} m "
    f"({overall['unrefined_mae_improvement_percent']:+.2f}%)"
)

print(
    f"RMSE: "
    f"{overall['midas_unrefined_rmse']:.6f}"
    f" -> "
    f"{overall['agbr_unrefined_rmse']:.6f} m "
    f"({overall['unrefined_rmse_improvement_percent']:+.2f}%)"
)

print(
    f"AbsRel: "
    f"{overall['midas_unrefined_absrel']:.6f}"
    f" -> "
    f"{overall['agbr_unrefined_absrel']:.6f} "
    f"({overall['unrefined_absrel_improvement_percent']:+.2f}%)"
)


print("\n==============================================")
print("FILES SAVED")
print("==============================================")

print(
    f"Frame-level:\n{frame_csv}"
)

print(
    f"\nScene-level:\n{scene_csv}"
)

print(
    f"\nOverall:\n{overall_csv}"
)

print("\n==============================================")
print("EVALUATION COMPLETE")
print("==============================================")