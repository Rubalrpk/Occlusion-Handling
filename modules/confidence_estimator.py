import cv2
import numpy as np


class ConfidenceEstimator:
    """Estimates a per-pixel confidence map for a MiDaS depth prediction,
    without requiring model retraining or a second network.

    Combines two cheap, CPU-friendly uncertainty signals:

      1. Local gradient magnitude — MiDaS's own documented weakness is
         boundary blur and thin-structure failure, i.e. errors concentrate
         where the depth map changes sharply. High local gradient -> lower
         confidence.

      2. Multi-scale disagreement — the source frame is deliberately
         degraded (downsampled then upsampled) and re-run through the SAME
         fixed-size MiDaS pipeline. Large disagreement between the normal
         and degraded predictions at a given pixel indicates the depth
         estimate there is unstable / scale-sensitive -> lower confidence.

         Note: this degrades the INPUT IMAGE, not the model's input tensor
         shape — MiDaS ONNX exports are typically fixed-shape (e.g. this
         model expects exactly 256x256), so running the model itself at a
         different resolution is not safe to assume will work.

    Output confidence is in [0, 1], where 1 = high confidence.
    """

    def __init__(self, depth_estimator, gradient_weight=0.5, multiscale_weight=0.5,
                 downscale_factor=0.4, gradient_blur_ksize=5):
        """
        Args:
            depth_estimator: an already-constructed DepthEstimator instance.
                Reused rather than re-loading a second ONNX session, since
                the model itself doesn't change between the two signals.
            gradient_weight, multiscale_weight: combination weights for the
                two uncertainty signals. Should sum to 1.0 but not enforced,
                so you can experiment.
            downscale_factor: how aggressively to degrade the frame for the
                multi-scale signal. Lower = more aggressive degradation =
                stronger disagreement signal, but also noisier.
            gradient_blur_ksize: Gaussian blur kernel applied to the raw
                gradient magnitude before normalizing, to avoid single-pixel
                noise dominating the uncertainty map.
        """
        self.depth_estimator = depth_estimator
        self.gradient_weight = gradient_weight
        self.multiscale_weight = multiscale_weight
        self.downscale_factor = downscale_factor
        self.gradient_blur_ksize = gradient_blur_ksize

    @staticmethod
    def _normalize01(arr):
        arr = arr.astype(np.float32)
        a_min, a_max = arr.min(), arr.max()
        denom = a_max - a_min
        if denom < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - a_min) / denom

    def _gradient_uncertainty(self, depth):
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        k = self.gradient_blur_ksize
        if k > 1:
            if k % 2 == 0:
                k += 1  # GaussianBlur requires an odd kernel size
            grad_mag = cv2.GaussianBlur(grad_mag, (k, k), 0)

        return self._normalize01(grad_mag)

    def _multiscale_disagreement(self, frame, depth_normal):
        h, w = frame.shape[:2]
        small_w = max(1, int(w * self.downscale_factor))
        small_h = max(1, int(h * self.downscale_factor))

        degraded = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)

        depth_degraded = self.depth_estimator.predict(degraded)

        d1 = self._normalize01(depth_normal)
        d2 = self._normalize01(depth_degraded)

        disagreement = np.abs(d1 - d2)
        return self._normalize01(disagreement)

    def estimate(self, frame, depth=None):
        """Compute a confidence map for the given frame.

        Args:
            frame: BGR frame (as read by cv2.imread / VideoCapture).
            depth: optionally pass an already-computed depth map for this
                   frame to avoid a redundant MiDaS call. If None, this
                   method runs depth_estimator.predict(frame) itself.

        Returns:
            (confidence, depth) — confidence is a float32 array in [0, 1]
            matching the frame's (H, W), 1 = high confidence. depth is the
            raw depth map used (useful if you passed None and want it
            downstream without recomputing).
        """
        if depth is None:
            depth = self.depth_estimator.predict(frame)

        grad_unc = self._gradient_uncertainty(depth)
        ms_unc = self._multiscale_disagreement(frame, depth)

        uncertainty = self.gradient_weight * grad_unc + self.multiscale_weight * ms_unc
        uncertainty = np.clip(uncertainty, 0.0, 1.0)

        confidence = 1.0 - uncertainty
        return confidence.astype(np.float32), depth

    def visualize(self, confidence):
        """Colorize a confidence map for display: red = low confidence,
        blue/green = high confidence (using COLORMAP_JET convention)."""
        vis = (confidence * 255).astype(np.uint8)
        return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    def low_confidence_mask(self, confidence, threshold=0.5):
        """Binary mask of pixels below a confidence threshold — this is the
        direct input to the region-selection stage next in the pipeline."""
        return (confidence < threshold).astype(np.uint8)


DEFAULT_MODEL_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\models\midas\midas_v21_small_256.onnx"
DEFAULT_CAMERA_INDEX = 0

if __name__ == "__main__":
    import sys
    import time
    import os
    from depth_estimation import DepthEstimator

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

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("Live confidence estimation running. Press 'q' or ESC to quit.")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed — camera disconnected?")
                break

            t0 = time.time()
            confidence, depth = confidence_model.estimate(frame)
            infer_ms = (time.time() - t0) * 1000

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            depth_normalized = depth_model.normalize_depth(depth)
            depth_vis = depth_model.colorize_depth(depth_normalized)
            confidence_vis = confidence_model.visualize(confidence)

            # Scale each panel down for display so frame+depth+confidence
            # together fit on screen regardless of webcam resolution.
            # This only affects what's shown in the window, not the
            # underlying computation above.
            display_width = 480
            panels = []
            for panel in (frame, depth_vis, confidence_vis):
                h, w = panel.shape[:2]
                scale = display_width / w
                resized = cv2.resize(panel, (display_width, int(h * scale)))
                panels.append(resized)
            combined = np.hstack(panels)

            label = f"FPS: {fps:.1f}  |  Depth+Confidence: {infer_ms:.1f} ms"
            cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.imshow("Live Confidence (frame | depth | confidence) - q/ESC to quit", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()