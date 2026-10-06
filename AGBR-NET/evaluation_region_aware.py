import os
import cv2
import numpy as np

from modules.depth_estimation import DepthEstimator
from modules.confidence_estimator import ConfidenceEstimator
from modules.region_selector import RegionSelector
from modules.gan_refinement import GANRefiner


# ============================================================
# PATHS
# ============================================================

MIDAS_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\models\midas\midas_v21_small_256.onnx"

GAN_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\models\gan\boundary_refinement_gan.onnx"

RGB_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\scene_a\keyframes\images\205892067862.jpg"

LIDAR_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\scene_a\keyframes\depth\205892067862.png"

OUTPUT_DIR = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\outputs\region_aware_test"


# ============================================================
# METRIC FUNCTIONS
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
        np.abs(error) / np.maximum(g, 1e-6)
    )

    return mae, rmse, abs_rel, len(g)


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
            "No valid pixels available for scale alignment."
        )

    # --------------------------------------------------------
    # DIRECT ORIENTATION
    # --------------------------------------------------------

    denom_direct = np.sum(p * p)

    if denom_direct > 1e-12:

        scale_direct = (
            np.sum(p * g)
            / denom_direct
        )

        aligned_direct = (
            scale_direct * p
        )

        mae_direct = np.mean(
            np.abs(aligned_direct - g)
        )

    else:

        scale_direct = 0.0
        mae_direct = np.inf

    # --------------------------------------------------------
    # INVERTED ORIENTATION
    # --------------------------------------------------------

    p_inv = prediction.max() - prediction

    p_inv_valid = p_inv[valid].astype(
        np.float64
    )

    denom_inv = np.sum(
        p_inv_valid * p_inv_valid
    )

    if denom_inv > 1e-12:

        scale_inv = (
            np.sum(p_inv_valid * g)
            / denom_inv
        )

        aligned_inv = (
            scale_inv * p_inv_valid
        )

        mae_inv = np.mean(
            np.abs(aligned_inv - g)
        )

    else:

        scale_inv = 0.0
        mae_inv = np.inf

    # --------------------------------------------------------
    # SELECT BEST ORIENTATION
    # --------------------------------------------------------

    if mae_direct <= mae_inv:

        aligned_full = (
            scale_direct * prediction
        )

        orientation = "direct"
        scale = scale_direct

    else:

        aligned_full = (
            scale_inv
            * (prediction.max() - prediction)
        )

        orientation = "inverted"
        scale = scale_inv

    return (
        aligned_full.astype(np.float32),
        orientation,
        scale
    )


def improvement(before, after):

    if before == 0 or np.isnan(before):
        return 0.0

    return (
        (before - after)
        / before
        * 100.0
    )


def print_comparison(title, baseline, refined):

    print("\n----------------------------------------")
    print(title)
    print("----------------------------------------")

    print(
        f"Pixels: {baseline[3]:,}"
    )

    print(
        f"MAE    : "
        f"{baseline[0]:.6f} -> "
        f"{refined[0]:.6f} m"
    )

    print(
        f"RMSE   : "
        f"{baseline[1]:.6f} -> "
        f"{refined[1]:.6f} m"
    )

    print(
        f"AbsRel : "
        f"{baseline[2]:.6f} -> "
        f"{refined[2]:.6f}"
    )

    print(
        f"MAE improvement: "
        f"{improvement(baseline[0], refined[0]):+.2f}%"
    )

    print(
        f"RMSE improvement: "
        f"{improvement(baseline[1], refined[1]):+.2f}%"
    )

    print(
        f"AbsRel improvement: "
        f"{improvement(baseline[2], refined[2]):+.2f}%"
    )


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    MIDAS_PATH,
    GAN_PATH,
    RGB_PATH,
    LIDAR_PATH
]:

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD RGB
# ============================================================

print("\n========================================")
print("AGBR REGION-AWARE LiDAR EVALUATION")
print("========================================")

frame = cv2.imread(
    RGB_PATH
)

if frame is None:

    raise RuntimeError(
        "Could not read RGB image."
    )

print(
    f"RGB: {frame.shape}"
)


# ============================================================
# LOAD POLYCAM LiDAR DEPTH
# ============================================================

lidar_raw = cv2.imread(
    LIDAR_PATH,
    cv2.IMREAD_UNCHANGED
)

if lidar_raw is None:

    raise RuntimeError(
        "Could not read LiDAR depth."
    )

# Polycam depth = millimeters
lidar_depth = (
    lidar_raw.astype(np.float32)
    / 1000.0
)

print(
    f"LiDAR: {lidar_depth.shape}"
)

print(
    f"LiDAR dtype: {lidar_raw.dtype}"
)

valid_lidar_values = lidar_depth[
    lidar_depth > 0
]

print(
    f"LiDAR range: "
    f"{valid_lidar_values.min():.3f} -> "
    f"{valid_lidar_values.max():.3f} m"
)


# ============================================================
# LOAD MODELS
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


# ============================================================
# 1. MiDaS
# ============================================================

print("\n[1] Running MiDaS...")

midas_depth = depth_model.predict(
    frame
)

print(
    f"MiDaS shape: {midas_depth.shape}"
)

print(
    f"MiDaS range: "
    f"{midas_depth.min():.4f} -> "
    f"{midas_depth.max():.4f}"
)


# ============================================================
# 2. CONFIDENCE
# ============================================================

print("\n[2] Computing confidence...")

confidence, _ = confidence_model.estimate(
    frame,
    depth=midas_depth
)

print(
    f"Mean confidence: "
    f"{confidence.mean():.4f}"
)


# ============================================================
# 3. REGION SELECTION
# ============================================================

print(
    "\n[3] Selecting adaptive regions..."
)

boxes = selector.select_regions(
    confidence
)

coverage = selector.coverage_fraction(
    boxes,
    frame.shape
)

print(
    f"Regions: {len(boxes)}"
)

print(
    f"GAN coverage: "
    f"{coverage * 100:.2f}%"
)

for i, box in enumerate(boxes):

    x0, y0, x1, y1 = box

    print(
        f"  Region {i + 1}: "
        f"({x0},{y0}) -> ({x1},{y1})"
    )


# ============================================================
# 4. BUILD FULL-RES REFINED REGION MASK
# ============================================================

H, W = frame.shape[:2]

refined_mask_full = np.zeros(
    (H, W),
    dtype=np.uint8
)

for x0, y0, x1, y1 in boxes:

    refined_mask_full[
        y0:y1,
        x0:x1
    ] = 1


# ============================================================
# 5. GAN REFINEMENT
# ============================================================

print(
    "\n[4] Running Boundary Refinement GAN..."
)

patches = selector.extract_patches(
    frame,
    midas_depth,
    boxes
)

refined_patches = gan.refine_all(
    patches
)

print(
    f"GAN refined regions: "
    f"{len(refined_patches)}"
)


# ============================================================
# 6. MERGE
# ============================================================

print(
    "\n[5] Creating final AGBR depth..."
)

agbr_depth = selector.merge_patches(
    midas_depth,
    refined_patches
)


# ============================================================
# 7. RESIZE TO POLYCAM LiDAR RESOLUTION
# ============================================================

target_h, target_w = lidar_depth.shape

midas_eval = cv2.resize(
    midas_depth,
    (target_w, target_h),
    interpolation=cv2.INTER_AREA
)

agbr_eval = cv2.resize(
    agbr_depth,
    (target_w, target_h),
    interpolation=cv2.INTER_AREA
)

refined_mask = cv2.resize(
    refined_mask_full,
    (target_w, target_h),
    interpolation=cv2.INTER_NEAREST
).astype(bool)


print(
    "\nEvaluation resolution:"
    f" {target_w} x {target_h}"
)


# ============================================================
# 8. VALID LiDAR MASK
# ============================================================

lidar_valid = (
    np.isfinite(lidar_depth)
    &
    (lidar_depth > 0)
)


# ============================================================
# 9. EVALUATION MASKS
# ============================================================

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


print("\n========================================")
print("EVALUATION MASKS")
print("========================================")

print(
    f"Global pixels: "
    f"{global_mask.sum():,}"
)

print(
    f"Refined-region pixels: "
    f"{refined_eval_mask.sum():,}"
)

print(
    f"Unrefined pixels: "
    f"{unrefined_eval_mask.sum():,}"
)


# ============================================================
# 10. CONTROLLED SCALE ALIGNMENT
# ============================================================

print(
    "\n[6] Controlled scale alignment..."
)

# Fit scale ONLY using MiDaS.
#
# AGBR receives exactly the same transformation.
#
# This is a controlled before-vs-after experiment.

midas_aligned, midas_orientation, midas_scale = (
    align_prediction(
        midas_eval,
        lidar_depth,
        global_mask
    )
)

print(
    f"MiDaS orientation: "
    f"{midas_orientation}"
)

print(
    f"MiDaS scale: "
    f"{midas_scale:.6f}"
)


# ------------------------------------------------------------
# APPLY SAME TRANSFORMATION TO AGBR
# ------------------------------------------------------------

if midas_orientation == "direct":

    agbr_aligned = (
        midas_scale
        * agbr_eval
    )

else:

    agbr_aligned = (
        midas_scale
        * (
            midas_eval.max()
            - agbr_eval
        )
    )

agbr_aligned = agbr_aligned.astype(
    np.float32
)

print(
    "AGBR uses the SAME scale "
    "and orientation as MiDaS."
)


# ============================================================
# 11. GLOBAL METRICS
# ============================================================

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


# ============================================================
# 12. REFINED REGION METRICS
# ============================================================

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


# ============================================================
# 13. UNREFINED REGION METRICS
# ============================================================

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


# ============================================================
# 14. RESULTS
# ============================================================

print("\n========================================")
print("RESULTS")
print("========================================")

print_comparison(
    "GLOBAL",
    midas_global,
    agbr_global
)

print_comparison(
    "REFINED GAN REGIONS",
    midas_refined,
    agbr_refined
)

print_comparison(
    "UNREFINED REGIONS",
    midas_unrefined,
    agbr_unrefined
)


# ============================================================
# 15. GAN CHANGE ANALYSIS
# ============================================================

depth_change = np.abs(
    agbr_depth.astype(np.float32)
    -
    midas_depth.astype(np.float32)
)

changed_pixels = np.sum(
    depth_change > 1e-6
)

total_pixels = depth_change.size

changed_percentage = (
    changed_pixels
    / total_pixels
    * 100.0
)

print("\n========================================")
print("GAN CHANGE")
print("========================================")

print(
    f"Changed pixels: "
    f"{changed_pixels:,} / "
    f"{total_pixels:,}"
)

print(
    f"Changed percentage: "
    f"{changed_percentage:.2f}%"
)

print(
    f"Mean depth change: "
    f"{depth_change.mean():.6f}"
)

print(
    f"Maximum depth change: "
    f"{depth_change.max():.6f}"
)


# ============================================================
# 16. SAVE VISUALIZATIONS
# ============================================================

print("\nSaving outputs...")


# MiDaS visualization
midas_vis = depth_model.colorize_depth(
    depth_model.normalize_depth(
        midas_depth
    )
)


# AGBR visualization
agbr_vis = depth_model.colorize_depth(
    depth_model.normalize_depth(
        agbr_depth
    )
)


# Confidence
confidence_vis = confidence_model.visualize(
    confidence
)


# Selected regions
region_vis = selector.visualize(
    frame,
    boxes
)


# GAN difference
difference_vis = cv2.normalize(
    depth_change,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

difference_vis = cv2.applyColorMap(
    difference_vis,
    cv2.COLORMAP_JET
)


cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "01_rgb.jpg"
    ),
    frame
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "02_midas_depth.jpg"
    ),
    midas_vis
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "03_confidence.jpg"
    ),
    confidence_vis
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "04_selected_regions.jpg"
    ),
    region_vis
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "05_agbr_depth.jpg"
    ),
    agbr_vis
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "06_gan_difference.jpg"
    ),
    difference_vis
)

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "07_refined_region_mask.png"
    ),
    (
        refined_mask.astype(np.uint8)
        * 255
    )
)


# ============================================================
# 17. SAVE RESULTS TO TXT
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "region_results.txt"
)

with open(
    results_file,
    "w"
) as f:

    f.write(
        "AGBR REGION-AWARE LiDAR EVALUATION\n"
    )

    f.write(
        "===================================\n\n"
    )

    f.write(
        f"Regions: {len(boxes)}\n"
    )

    f.write(
        f"GAN coverage: "
        f"{coverage * 100:.4f}%\n"
    )

    f.write(
        f"Changed percentage: "
        f"{changed_percentage:.4f}%\n\n"
    )

    f.write(
        f"Scale orientation: "
        f"{midas_orientation}\n"
    )

    f.write(
        f"Common scale: "
        f"{midas_scale:.8f}\n\n"
    )

    for name, baseline, refined in [

        (
            "GLOBAL",
            midas_global,
            agbr_global
        ),

        (
            "REFINED GAN REGIONS",
            midas_refined,
            agbr_refined
        ),

        (
            "UNREFINED REGIONS",
            midas_unrefined,
            agbr_unrefined
        )

    ]:

        f.write(
            f"{name}\n"
        )

        f.write(
            f"Pixels: "
            f"{baseline[3]}\n"
        )

        f.write(
            f"MiDaS MAE: "
            f"{baseline[0]:.8f}\n"
        )

        f.write(
            f"AGBR MAE: "
            f"{refined[0]:.8f}\n"
        )

        f.write(
            f"MiDaS RMSE: "
            f"{baseline[1]:.8f}\n"
        )

        f.write(
            f"AGBR RMSE: "
            f"{refined[1]:.8f}\n"
        )

        f.write(
            f"MiDaS AbsRel: "
            f"{baseline[2]:.8f}\n"
        )

        f.write(
            f"AGBR AbsRel: "
            f"{refined[2]:.8f}\n"
        )

        f.write(
            f"MAE improvement: "
            f"{improvement(baseline[0], refined[0]):.4f}%\n"
        )

        f.write(
            f"RMSE improvement: "
            f"{improvement(baseline[1], refined[1]):.4f}%\n"
        )

        f.write(
            f"AbsRel improvement: "
            f"{improvement(baseline[2], refined[2]):.4f}%\n\n"
        )


print(
    f"\nResults saved to:\n"
    f"{results_file}"
)

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")