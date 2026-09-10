import cv2

from modules.depth_estimation import DepthEstimator
from modules.segmentation import SegmentationModel

# -----------------------------
# Model Paths
# -----------------------------
DEPTH_MODEL_PATH = "models/midas/midas_v21_small_256.onnx"
SEG_MODEL_PATH = "models/u2net/u2net.onnx"

# -----------------------------
# Test Image
# -----------------------------
IMAGE_PATH = "datasets/test_images/test.webp"

# -----------------------------
# Load Models
# -----------------------------
depth_model = DepthEstimator(DEPTH_MODEL_PATH)
segmenter = SegmentationModel(SEG_MODEL_PATH)

# -----------------------------
# Read Image
# -----------------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise Exception("Image could not be loaded.")

# -----------------------------
# Depth Estimation
# -----------------------------
depth = depth_model.predict(image)
depth = depth_model.normalize_depth(depth)

# -----------------------------
# Semantic Segmentation
# -----------------------------
mask = segmenter.predict(image)
mask = segmenter.postprocess(mask)

# -----------------------------
# Display Results
# -----------------------------
cv2.imshow("Original Image", image)
cv2.imshow("Depth Map", depth)
cv2.imshow("Segmentation Mask", mask)

# -----------------------------
# Save Results
# -----------------------------
cv2.imwrite("outputs/depth.png", depth)
cv2.imwrite("outputs/mask.png", mask)

# -----------------------------
# Exit
# -----------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()