"""V2 adaptive region selection for ABGR-Net.

Implements the validated gradient-guided selection sequence:
Gaussian smoothing -> Sobel gradient magnitude -> P90 threshold ->
3x3 morphological opening -> 8-connected components -> minimum area
filter -> largest-N regions. This is intentionally separate from the
legacy RegionSelector, whose fixed confidence-threshold logic is retained.
"""
import cv2
import numpy as np


class ABGRV2RegionSelector:
    def __init__(
        self,
        percentile=90.0,
        min_region_area=30,
        max_regions=8,
        context_margin=12,
        gaussian_sigma=1.0,
    ):
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be between 0 and 100")
        if min_region_area < 1 or max_regions < 1 or context_margin < 0:
            raise ValueError("Invalid region selector configuration")

        self.percentile = float(percentile)
        self.min_region_area = int(min_region_area)
        self.max_regions = int(max_regions)
        self.context_margin = int(context_margin)
        self.gaussian_sigma = float(gaussian_sigma)

    @staticmethod
    def gradient_map(depth):
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth must be a non-empty 2D array")

        smoothed = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.0)
        gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    def estimate(self, depth):
        """Return normalized gradient, heuristic confidence, and binary mask."""
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth must be a non-empty 2D array")

        smoothed = cv2.GaussianBlur(
            depth, (0, 0), sigmaX=self.gaussian_sigma
        )
        gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx * gx + gy * gy)

        g_min = float(gradient.min())
        g_max = float(gradient.max())
        normalized = (gradient - g_min) / (g_max - g_min + 1e-6)
        confidence = np.clip(1.0 - normalized, 0.0, 1.0)

        threshold = float(np.percentile(gradient, self.percentile))
        mask = (gradient >= threshold).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return normalized.astype(np.float32), confidence.astype(np.float32), mask, threshold

    def select_regions(self, depth):
        normalized, confidence, mask, threshold = self.estimate(depth)
        height, width = depth.shape[:2]

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        candidates = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_region_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            candidates.append((x, y, w, h, area))

        candidates.sort(key=lambda item: item[4], reverse=True)
        boxes = []
        for x, y, w, h, _area in candidates[:self.max_regions]:
            x0 = max(0, x - self.context_margin)
            y0 = max(0, y - self.context_margin)
            x1 = min(width, x + w + self.context_margin)
            y1 = min(height, y + h + self.context_margin)
            boxes.append((x0, y0, x1, y1))

        diagnostics = {
            "gradient": normalized,
            "confidence": confidence,
            "mask": mask,
            "threshold": threshold,
            "candidate_count": len(candidates),
        }
        return boxes, diagnostics

    @staticmethod
    def coverage_fraction(boxes, frame_shape):
        height, width = frame_shape[:2]
        if not boxes:
            return 0.0
        union = np.zeros((height, width), dtype=np.uint8)
        for x0, y0, x1, y1 in boxes:
            union[y0:y1, x0:x1] = 1
        return float(union.sum()) / float(height * width)

    @staticmethod
    def visualize(frame, boxes):
        output = frame.copy()
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(output, (x0, y0), (x1, y1), (0, 255, 255), 2)
        return output
