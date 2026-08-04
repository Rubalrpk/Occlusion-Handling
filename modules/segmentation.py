import cv2
import numpy as np
import onnxruntime as ort


class SegmentationModel:
    """ONNX Runtime wrapper for U²-Net salient object segmentation.

    predict() returns a binary foreground mask resized back to the
    original input frame's dimensions, ready for pixel-wise fusion
    with a depth map of the same size.

    Includes two fixes for known U2-Net weaknesses on live human subjects:
      - Letterbox (aspect-preserving) preprocessing instead of a direct
        squish-resize, so body proportions aren't distorted before the
        model sees them.
      - Optional temporal EMA smoothing of the soft mask across frames,
        to reduce frame-to-frame flicker from borderline pixels crossing
        the threshold. Enabled by default.

    Note: transparent/thin structures (glasses lenses, fine hair, cables)
    are a known failure mode of U2-Net itself, not a preprocessing issue —
    no amount of pre/post-processing here fully fixes it. That gap is the
    empirical justification for a boundary-refinement stage on top of this.
    """

    def __init__(self, model_path, input_size=320, threshold=0.5,
                 temporal_smoothing=True, temporal_alpha=0.6,
                 morphology_cleanup=True):
        self.input_size = input_size
        self.threshold = threshold
        self.temporal_smoothing = temporal_smoothing
        self.temporal_alpha = temporal_alpha  # weight on the NEW frame; higher = less smoothing/lag
        self.morphology_cleanup = morphology_cleanup

        self._prev_soft_mask = None  # persists across predict() calls for temporal smoothing

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("U2-Net loaded successfully.")

    def reset_temporal_state(self):
        """Call this if the camera cuts to a completely different scene —
        otherwise the smoothing will blend the old scene into the new one
        for a few frames."""
        self._prev_soft_mask = None

    def _letterbox(self, image):
        """Pad the frame to square (preserving aspect ratio) before resizing
        to the model's input size, instead of squishing it directly. Returns
        the padded-square image plus enough info to undo the padding later.
        """
        h, w = image.shape[:2]
        side = max(h, w)

        pad_top = (side - h) // 2
        pad_bottom = side - h - pad_top
        pad_left = (side - w) // 2
        pad_right = side - w - pad_left

        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded, (pad_top, pad_bottom, pad_left, pad_right, side)

    def preprocess(self, image):
        orig_h, orig_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded, pad_info = self._letterbox(rgb)
        resized = cv2.resize(padded, (self.input_size, self.input_size))

        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, (orig_h, orig_w), pad_info

    def predict(self, image, binary=True):
        """Run inference and return a mask resized to the input frame's
        original dimensions.

        Args:
            image: BGR frame (as read by cv2.imread / VideoCapture).
            binary: if True, returns a {0,1} uint8 mask (foreground=1).
                    if False, returns a normalized [0,255] soft mask.
        """
        tensor, orig_size, pad_info = self.preprocess(image)

        raw_output = self.session.run(
            [self.output_name],
            {self.input_name: tensor}
        )[0]

        soft_mask = self._postprocess(raw_output, orig_size, pad_info)

        if self.temporal_smoothing:
            soft_mask = self._apply_temporal_smoothing(soft_mask)

        if binary:
            _, mask = cv2.threshold(
                soft_mask, int(self.threshold * 255), 1, cv2.THRESH_BINARY
            )
            mask = mask.astype(np.uint8)
            if self.morphology_cleanup:
                mask = self._cleanup_mask(mask)
            return mask

        return soft_mask

    def _apply_temporal_smoothing(self, soft_mask):
        if self._prev_soft_mask is None or self._prev_soft_mask.shape != soft_mask.shape:
            self._prev_soft_mask = soft_mask.astype(np.float32)
            return soft_mask

        blended = (self.temporal_alpha * soft_mask.astype(np.float32) +
                   (1 - self.temporal_alpha) * self._prev_soft_mask)
        self._prev_soft_mask = blended
        return blended.astype(np.uint8)

    def _cleanup_mask(self, mask):
        """Fill small holes (e.g. the gap U2-Net leaves for glasses lenses)
        and drop small flicker specks, keeping only the largest connected
        blob — a cheap partial mitigation, not a fix for the underlying
        transparent-object blind spot."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        if num_labels <= 1:
            return closed  # nothing but background found

        # Label 0 is background; pick the largest non-background component.
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleaned = (labels == largest_label).astype(np.uint8)
        return cleaned

    def _postprocess(self, output, orig_size, pad_info):
        mask = np.squeeze(output)

        mask_min, mask_max = mask.min(), mask.max()
        denom = mask_max - mask_min
        if denom < 1e-6:
            mask = np.zeros_like(mask, dtype=np.float32)
        else:
            mask = (mask - mask_min) / denom

        mask = (mask * 255).astype(np.uint8)

        # Undo the letterbox: resize up to the padded-square size, then crop
        # back to the original (unpadded) frame dimensions.
        pad_top, pad_bottom, pad_left, pad_right, side = pad_info
        mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_LINEAR)

        orig_h, orig_w = orig_size
        mask = mask[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w]

        return mask

    def save_mask(self, mask, path):
        if mask.max() <= 1:
            mask = mask * 255
        cv2.imwrite(path, mask)


DEFAULT_MODEL_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\models\u2net\u2net.onnx"
DEFAULT_CAMERA_INDEX = 0

if __name__ == "__main__":
    import sys
    import time
    import os

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAMERA_INDEX

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"U2-Net model not found at: {model_path}\n"
            "Either fix DEFAULT_MODEL_PATH at the top of this file, or pass the "
            "correct path as the first command-line argument."
        )

    model = SegmentationModel(model_path)


    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check that the webcam is connected and not in use by another app."
        )

    print("Live segmentation running. Press 'q' or ESC to quit.")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed — camera disconnected?")
                break

            t0 = time.time()
            mask = model.predict(frame, binary=True)
            infer_ms = (time.time() - t0) * 1000

            # Smoothed FPS over the full loop (grab + inference + display)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # Visualize: binary mask scaled to 0/255, side-by-side with the live frame
            mask_vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([frame, mask_vis])

            label = f"FPS: {fps:.1f}  |  U2-Net inference: {infer_ms:.1f} ms"
            cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.imshow("Live Segmentation (frame | mask) — q/ESC to quit", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()