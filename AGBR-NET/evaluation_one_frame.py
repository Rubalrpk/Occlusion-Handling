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

OUTPUT_DIR = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\outputs\one_frame_test"


# ============================================================
# CHECK FILES
# ============================================================

for path in [MIDAS_PATH, GAN_PATH, RGB_PATH, LIDAR_PATH]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"\nFile not found:\n{path}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD RGB
# ============================================================

frame = cv2.imread(RGB_PATH)

if frame is None:
    raise RuntimeError(f"Could not read RGB image:\n{RGB_PATH}")

print("\n========================================")
print("AGBR + LiDAR ONE-FRAME EVALUATION")
print("========================================")

print(f"RGB shape: {frame.shape}")


# ============================================================
# LOAD LIDAR DEPTH
# ============================================================

lidar_depth_raw = cv2.imread(
    LIDAR_PATH,
    cv2.IMREAD_UNCHANGED
)

if lidar_depth_raw is None:
    raise RuntimeError(f"Could not read LiDAR depth:\n{LIDAR_PATH}")

print(f"LiDAR depth shape: {lidar_depth_raw.shape}")
print(f"LiDAR dtype: {lidar_depth_raw.dtype}")

# Polycam depth is stored in millimeters.
lidar_depth = lidar_depth_raw.astype(np.float32) / 1000.0

print(
    f"LiDAR depth range: "
    f"{lidar_depth[lidar_depth > 0].min():.3f} -> "
    f"{lidar_depth.max():.3f} m"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading MiDaS...")
depth_model = DepthEstimator(MIDAS_PATH)

print("Loading confidence estimator...")
confidence_model = ConfidenceEstimator(depth_model)

print("Loading region selector...")
selector = RegionSelector()

print("Loading GAN...")
gan = GANRefiner(GAN_PATH)


# ============================================================
# 1. MiDaS
# ============================================================

print("\n[1] Running MiDaS...")

midas_depth = depth_model.predict(frame)

print(f"MiDaS shape: {midas_depth.shape}")
print(
    f"MiDaS range: "
    f"{midas_depth.min():.4f} -> {midas_depth.max():.4f}"
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

print("\n[3] Selecting adaptive regions...")

boxes = selector.select_regions(confidence)

coverage = selector.coverage_fraction(
    boxes,
    frame.shape
)

print(f"Regions selected: {len(boxes)}")
print(f"GAN coverage: {coverage * 100:.2f}%")


for i, box in enumerate(boxes):

    x0, y0, x1, y1 = box

    print(
        f"  Region {i + 1}: "
        f"({x0},{y0}) -> ({x1},{y1})"
    )


# ============================================================
# 4. GAN
# ============================================================

print("\n[4] Running Boundary Refinement GAN...")

patches = selector.extract_patches(
    frame,
    midas_depth,
    boxes
)

refined_patches = gan.refine_all(patches)

print(
    f"GAN refined regions: "
    f"{len(refined_patches)}"
)


# ============================================================
# 5. MERGE
# ============================================================

print("\n[5] Creating final AGBR depth...")

agbr_depth = selector.merge_patches(
    midas_depth,
    refined_patches
)


# ============================================================
# 6. RESIZE TO LIDAR RESOLUTION
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

print(
    f"\nEvaluation resolution: "
    f"{target_w} x {target_h}"
)


# ============================================================
# 7. VALID LIDAR MASK
# ============================================================

valid_mask = (
    np.isfinite(lidar_depth)
    &
    (lidar_depth > 0)
)

print(
    f"Valid LiDAR pixels: "
    f"{valid_mask.sum():,} / {valid_mask.size:,}"
)


# ============================================================
# 8. SCALE ALIGNMENT
# ============================================================

def scale_align_prediction(prediction, ground_truth, mask):
    """
    Align a relative depth prediction to metric LiDAR depth.

    Two orientations are tested:
      1. prediction itself
      2. inverted prediction

    The orientation giving lower MAE is selected.

    Then a least-squares scale is fitted.
    """

    p = prediction[mask].astype(np.float64)
    g = ground_truth[mask].astype(np.float64)

    # Remove invalid values
    valid = (
        np.isfinite(p)
        &
        np.isfinite(g)
        &
        (g > 0)
    )

    p = p[valid]
    g = g[valid]

    # ---------- orientation 1 ----------
    p1 = p.copy()

    denom1 = np.dot(p1, p1)

    if denom1 > 1e-12:
        scale1 = np.dot(p1, g) / denom1
        aligned1 = scale1 * p1

        mae1 = np.mean(np.abs(aligned1 - g))
    else:
        scale1 = 0.0
        aligned1 = np.zeros_like(g)
        mae1 = np.inf

    # ---------- orientation 2 ----------
    p2 = p.max() - p

    denom2 = np.dot(p2, p2)

    if denom2 > 1e-12:
        scale2 = np.dot(p2, g) / denom2
        aligned2 = scale2 * p2

        mae2 = np.mean(np.abs(aligned2 - g))
    else:
        scale2 = 0.0
        aligned2 = np.zeros_like(g)
        mae2 = np.inf

    if mae1 <= mae2:

        aligned = scale1 * prediction
        orientation = "direct"
        scale = scale1

    else:

        aligned = scale2 * (prediction.max() - prediction)
        orientation = "inverted"
        scale = scale2

    return aligned, orientation, scale


# ============================================================
# 9. METRICS
# ============================================================

def calculate_metrics(prediction, ground_truth, mask):

    p = prediction[mask]
    g = ground_truth[mask]

    valid = (
        np.isfinite(p)
        &
        np.isfinite(g)
        &
        (g > 0)
    )

    p = p[valid]
    g = g[valid]

    error = p - g

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    abs_rel = np.mean(
        np.abs(error) / np.maximum(g, 1e-6)
    )

    return mae, rmse, abs_rel, len(g)


# ============================================================
# 10. ALIGN MiDaS
# ============================================================

print("\n[6] Aligning MiDaS to LiDAR...")

midas_aligned, midas_orientation, midas_scale = (
    scale_align_prediction(
        midas_eval,
        lidar_depth,
        valid_mask
    )
)

midas_mae, midas_rmse, midas_absrel, n = (
    calculate_metrics(
        midas_aligned,
        lidar_depth,
        valid_mask
    )
)


# ============================================================
# 11. ALIGN AGBR
# ============================================================

print("[7] Aligning AGBR to LiDAR...")

agbr_aligned, agbr_orientation, agbr_scale = (
    scale_align_prediction(
        agbr_eval,
        lidar_depth,
        valid_mask
    )
)

agbr_mae, agbr_rmse, agbr_absrel, _ = (
    calculate_metrics(
        agbr_aligned,
        lidar_depth,
        valid_mask
    )
)


# ============================================================
# 12. IMPROVEMENT
# ============================================================

mae_improvement = (
    (midas_mae - agbr_mae)
    / max(midas_mae, 1e-12)
    * 100
)

rmse_improvement = (
    (midas_rmse - agbr_rmse)
    / max(midas_rmse, 1e-12)
    * 100
)

absrel_improvement = (
    (midas_absrel - agbr_absrel)
    / max(midas_absrel, 1e-12)
    * 100
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print("\nVALID PIXELS")
print(f"{n:,}")

print("\nMiDaS alignment:")
print(f"  Orientation : {midas_orientation}")
print(f"  Scale       : {midas_scale:.6f}")

print("\nAGBR alignment:")
print(f"  Orientation : {agbr_orientation}")
print(f"  Scale       : {agbr_scale:.6f}")


print("\n----------------------------------------")
print("MiDaS BASELINE")
print("----------------------------------------")

print(f"MAE    : {midas_mae:.6f} m")
print(f"RMSE   : {midas_rmse:.6f} m")
print(f"AbsRel : {midas_absrel:.6f}")


print("\n----------------------------------------")
print("AGBR")
print("----------------------------------------")

print(f"MAE    : {agbr_mae:.6f} m")
print(f"RMSE   : {agbr_rmse:.6f} m")
print(f"AbsRel : {agbr_absrel:.6f}")


print("\n----------------------------------------")
print("IMPROVEMENT")
print("----------------------------------------")

print(f"MAE    : {mae_improvement:+.2f}%")
print(f"RMSE   : {rmse_improvement:+.2f}%")
print(f"AbsRel : {absrel_improvement:+.2f}%")


# ============================================================
# 14. SAVE VISUALIZATIONS
# ============================================================

print("\nSaving outputs...")


midas_vis = depth_model.colorize_depth(
    depth_model.normalize_depth(midas_depth)
)

agbr_vis = depth_model.colorize_depth(
    depth_model.normalize_depth(agbr_depth)
)

confidence_vis = confidence_model.visualize(
    confidence
)

region_vis = selector.visualize(
    frame,
    boxes
)

difference = np.abs(
    agbr_depth.astype(np.float32)
    -
    midas_depth.astype(np.float32)
)

difference_vis = cv2.normalize(
    difference,
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
    os.path.join(OUTPUT_DIR, "01_rgb.jpg"),
    frame
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "02_midas_depth.jpg"),
    midas_vis
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "03_confidence.jpg"),
    confidence_vis
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "04_selected_regions.jpg"),
    region_vis
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "05_agbr_depth.jpg"),
    agbr_vis
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "06_gan_difference.jpg"),
    difference_vis
)


# ============================================================
# SAVE RAW EVALUATION ARRAYS
# ============================================================

np.save(
    os.path.join(OUTPUT_DIR, "midas_eval.npy"),
    midas_eval
)

np.save(
    os.path.join(OUTPUT_DIR, "agbr_eval.npy"),
    agbr_eval
)

np.save(
    os.path.join(OUTPUT_DIR, "lidar_depth.npy"),
    lidar_depth
)


# ============================================================
# SAVE TEXT RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "results.txt"
)

with open(results_path, "w") as f:

    f.write("AGBR ONE-FRAME LiDAR EVALUATION\n")
    f.write("================================\n\n")

    f.write(f"RGB: {RGB_PATH}\n")
    f.write(f"LiDAR: {LIDAR_PATH}\n\n")

    f.write(f"Valid pixels: {n}\n\n")

    f.write("MiDaS\n")
    f.write(f"Orientation: {midas_orientation}\n")
    f.write(f"Scale: {midas_scale}\n")
    f.write(f"MAE: {midas_mae}\n")
    f.write(f"RMSE: {midas_rmse}\n")
    f.write(f"AbsRel: {midas_absrel}\n\n")

    f.write("AGBR\n")
    f.write(f"Orientation: {agbr_orientation}\n")
    f.write(f"Scale: {agbr_scale}\n")
    f.write(f"MAE: {agbr_mae}\n")
    f.write(f"RMSE: {agbr_rmse}\n")
    f.write(f"AbsRel: {agbr_absrel}\n\n")

    f.write("Improvement\n")
    f.write(f"MAE: {mae_improvement}%\n")
    f.write(f"RMSE: {rmse_improvement}%\n")
    f.write(f"AbsRel: {absrel_improvement}%\n")


print(
    f"\nResults saved to:\n{results_path}"
)

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")