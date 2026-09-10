import cv2
import numpy as np


class RegionSelector:
    """Converts a per-pixel confidence map into a small set of bounding-box
    regions to send to the boundary refinement GAN — this is the module
    that actually implements "adaptive" in your pipeline: instead of
    refining the whole frame, only low-confidence regions (typically a
    thin ring around occlusion boundaries) get the expensive treatment.

    Also provides the inverse operation: pasting refined patches back into
    the full depth map, with edge feathering so patch boundaries don't
    show up as visible seams in the final result.
    """

    def __init__(self, confidence_threshold=0.6, min_region_area=150,
                 dilation_kernel=15, context_margin=12, max_regions=8,
                 feather_px=8):
        """
        Args:
            confidence_threshold: pixels with confidence below this are
                considered "needs refinement".
            min_region_area: connected components smaller than this (in
                pixels, after dilation) are dropped as noise rather than
                sent to the GAN — avoids wasting a patch inference on a
                handful of stray pixels.
            dilation_kernel: size of the morphological dilation applied to
                the low-confidence mask before finding connected components.
                This merges nearby uncertain pixels into one region and
                gives the GAN a bit of surrounding context rather than a
                razor-thin selection.
            context_margin: extra pixels of padding added around each
                selected region's bounding box, so the GAN sees some
                confidently-known surrounding context, not just the
                uncertain pixels in isolation.
            max_regions: hard cap on how many regions get sent to the GAN
                per frame, prioritized by area (largest/most uncertain
                first). This is a safety valve — without it, a very noisy
                confidence map could fragment into dozens of tiny regions
                and erase your FPS gains from being "adaptive" at all.
            feather_px: width of the alpha feathering ramp used when
                pasting refined patches back into the depth map, to avoid
                hard rectangular seams.
        """
        self.confidence_threshold = confidence_threshold
        self.min_region_area = min_region_area
        self.dilation_kernel = dilation_kernel
        self.context_margin = context_margin
        self.max_regions = max_regions
        self.feather_px = feather_px

    def select_regions(self, confidence):
        """Returns a list of (x0, y0, x1, y1) boxes to refine, sorted by
        priority (largest/most-uncertain region first). Empty list means
        no refinement needed this frame."""
        H, W = confidence.shape[:2]

        low_conf_mask = (confidence < self.confidence_threshold).astype(np.uint8)

        if self.dilation_kernel > 1:
            k = self.dilation_kernel
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            low_conf_mask = cv2.dilate(low_conf_mask, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            low_conf_mask, connectivity=8
        )

        candidates = []
        for i in range(1, num_labels):  # label 0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_region_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            candidates.append((x, y, w, h, area))

        candidates.sort(key=lambda c: c[4], reverse=True)
        candidates = candidates[:self.max_regions]

        boxes = []
        for (x, y, w, h, _area) in candidates:
            x0 = max(0, x - self.context_margin)
            y0 = max(0, y - self.context_margin)
            x1 = min(W, x + w + self.context_margin)
            y1 = min(H, y + h + self.context_margin)
            boxes.append((x0, y0, x1, y1))

        return boxes

    def coverage_fraction(self, boxes, frame_shape):
        """Fraction of the total frame area covered by the selected boxes
        (accounting for overlap). This is the number your efficiency /
        Pareto-curve results section needs — how much of the frame
        actually gets the expensive GAN treatment, on average, per frame.
        """
        H, W = frame_shape[:2]
        if not boxes:
            return 0.0

        union_mask = np.zeros((H, W), dtype=np.uint8)
        for (x0, y0, x1, y1) in boxes:
            union_mask[y0:y1, x0:x1] = 1

        return float(union_mask.sum()) / float(H * W)

    def extract_patches(self, frame, depth, boxes):
        """Given selected boxes, extract the corresponding RGB and depth
        patches — this is what actually gets fed into the boundary
        refinement GAN next in the pipeline."""
        patches = []
        for (x0, y0, x1, y1) in boxes:
            rgb_patch = frame[y0:y1, x0:x1].copy()
            depth_patch = depth[y0:y1, x0:x1].copy()
            patches.append({
                "box": (x0, y0, x1, y1),
                "rgb": rgb_patch,
                "depth": depth_patch,
            })
        return patches

    def _feather_alpha(self, h, w):
        """Builds a soft-edged alpha mask (1 in the interior, ramping to 0
        at the border over feather_px pixels) so pasted patches blend
        smoothly instead of leaving a visible rectangular seam."""
        alpha = np.ones((h, w), dtype=np.float32)
        f = min(self.feather_px, h // 2, w // 2)
        if f < 1:
            return alpha

        ramp = np.linspace(0, 1, f, dtype=np.float32)
        alpha[:f, :] *= ramp[:, None]
        alpha[-f:, :] *= ramp[::-1][:, None]
        alpha[:, :f] *= ramp[None, :]
        alpha[:, -f:] *= ramp[::-1][None, :]
        return alpha

    def merge_patches(self, depth, refined_patches):
        """Pastes refined depth patches back into the full-frame depth map
        with feathered edges. refined_patches is a list of dicts matching
        extract_patches()'s format, but with a "depth" array that's been
        replaced by the GAN's refined output for that patch.

        If refined_patches is empty (nothing was selected this frame),
        returns the input depth map unchanged.
        """
        if not refined_patches:
            return depth.copy()

        output = depth.astype(np.float32).copy()

        for patch in refined_patches:
            x0, y0, x1, y1 = patch["box"]
            refined = patch["depth"].astype(np.float32)
            h, w = y1 - y0, x1 - x0

            if refined.shape[:2] != (h, w):
                refined = cv2.resize(refined, (w, h), interpolation=cv2.INTER_LINEAR)

            alpha = self._feather_alpha(h, w)
            original_region = output[y0:y1, x0:x1]
            output[y0:y1, x0:x1] = alpha * refined + (1 - alpha) * original_region

        return output

    def filter_regions_temporal(self, boxes, change_mask, min_change_fraction=0.05):
        """Splits confidence-selected boxes into two groups using a
        frame-to-frame change mask (from TemporalCache.compute_change_mask):

          - boxes that overlap enough actual pixel change to warrant a
            fresh GAN refinement this frame
          - boxes that are still low-confidence (still near a boundary)
            but haven't meaningfully changed since last frame, and can
            just reuse last frame's cached refined result instead

        This is the core of the temporal reuse mechanism: it's what lets
        the pipeline skip GAN inference on regions that are low-confidence
        but static, which is the common case for a mostly-still scene.

        Args:
            boxes: list of (x0, y0, x1, y1), as returned by select_regions().
            change_mask: binary (H, W) array from TemporalCache, 1=changed.
            min_change_fraction: minimum fraction of a box's pixels that
                must be flagged as changed for it to be re-refined rather
                than reused. Low values make reuse rare (safer, less
                speedup); high values make reuse aggressive (faster, more
                risk of missing a real but small change).

        Returns:
            (boxes_to_refine, boxes_to_reuse) — both lists of
            (x0, y0, x1, y1), partitioning the input boxes.
        """
        boxes_to_refine = []
        boxes_to_reuse = []

        for box in boxes:
            x0, y0, x1, y1 = box
            region_change = change_mask[y0:y1, x0:x1]
            if region_change.size == 0:
                continue

            change_fraction = float(region_change.mean())
            if change_fraction >= min_change_fraction:
                boxes_to_refine.append(box)
            else:
                boxes_to_reuse.append(box)

        return boxes_to_refine, boxes_to_reuse

    def visualize(self, frame, boxes):
        """Draws selected region boxes on a copy of the frame, for
        debugging / figures."""
        vis = frame.copy()
        for (x0, y0, x1, y1) in boxes:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
        return vis


DEFAULT_MODEL_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\models\midas\midas_v21_small_256.onnx"
DEFAULT_CAMERA_INDEX = 0

if __name__ == "__main__":
    import sys
    import time
    import os
    from depth_estimation import DepthEstimator
    from confidence_estimator import ConfidenceEstimator

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAMERA_INDEX

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"MiDaS model not found at: {model_path}\n"
            "Either fix DEFAULT_MODEL_PATH at the top of this file, or pass the "
            "correct path as the first command-line argument."
        )

    depth_model = DepthEstimator(model_path)
    confidence_model = ConfidenceEstimator(depth_model)
    region_selector = RegionSelector()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("Live region selection running. Press 'q' or ESC to quit.")

    prev_time = time.time()
    fps = 0.0
    avg_coverage = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed — camera disconnected?")
                break

            t0 = time.time()
            confidence, depth = confidence_model.estimate(frame)
            boxes = region_selector.select_regions(confidence)
            coverage = region_selector.coverage_fraction(boxes, frame.shape)
            infer_ms = (time.time() - t0) * 1000

            avg_coverage = 0.9 * avg_coverage + 0.1 * coverage

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            vis = region_selector.visualize(frame, boxes)

            label1 = f"FPS: {fps:.1f}  |  Stage time: {infer_ms:.1f} ms"
            label2 = f"Regions: {len(boxes)}  |  Coverage: {coverage*100:.1f}% (avg {avg_coverage*100:.1f}%)"
            cv2.putText(vis, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, label2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Live Region Selection (yellow boxes = sent to GAN) - q/ESC to quit", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()