import cv2
import numpy as np
import onnxruntime as ort


class DepthEstimator:
    """ONNX Runtime wrapper for MiDaS monocular depth estimation.

    predict() returns a relative depth map resized back to the original
    input frame's dimensions, so it aligns pixel-for-pixel with the
    SegmentationModel's output mask during occlusion reasoning.
    """

    def __init__(self, model_path, default_input_size=256, verbose=False):
        self.verbose = verbose

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Auto-detect the model's expected spatial input size rather than
        # trusting a hardcoded constant that can silently drift out of sync
        # with whatever .onnx file actually gets loaded.
        input_shape = self.session.get_inputs()[0].shape
        h = input_shape[2] if len(input_shape) == 4 and isinstance(input_shape[2], int) else default_input_size
        w = input_shape[3] if len(input_shape) == 4 and isinstance(input_shape[3], int) else default_input_size
        self.input_size = (w, h)  # cv2.resize expects (width, height)

        print(f"MiDaS model loaded successfully. Input size: {self.input_size}")

    def preprocess(self, image):
        """Convert BGR frame to the model's expected normalized NCHW tensor.
        Returns the tensor plus the original (H, W) so depth can be resized
        back after inference.
        """
        orig_h, orig_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)

        tensor = resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std

        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        return tensor, (orig_h, orig_w)

    def predict(self, image):
        """Run inference and return a relative depth map resized to the
        input frame's original (H, W). Values remain raw/relative
        (not normalized to 0-255) — use normalize_depth() for that.
        """
        input_tensor, orig_size = self.preprocess(image)

        raw_output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]

        depth = np.squeeze(raw_output).astype(np.float32)

        # Resize back to the original frame's (H, W) — required so depth
        # aligns pixel-for-pixel with the segmentation mask downstream.
        orig_h, orig_w = orig_size
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        if self.verbose:
            print(f"Depth shape: {depth.shape}, min: {depth.min():.4f}, max: {depth.max():.4f}")

        return depth

    def normalize_depth(self, depth):
        """Rescale relative depth to a 0-255 uint8 image for visualization
        or as input to colorize_depth(). Not needed for the raw occlusion
        comparison, which should use predict()'s raw output directly.
        """
        depth = depth - depth.min()
        denom = depth.max() + 1e-8
        depth = depth / denom
        depth = np.clip(depth * 255, 0, 255).astype(np.uint8)
        return depth

    def colorize_depth(self, depth_uint8):
        """Apply a color map. Expects a normalized uint8 depth map
        (i.e. the output of normalize_depth()), not raw predict() output.
        """
        if depth_uint8.dtype != np.uint8:
            raise ValueError(
                "colorize_depth expects a uint8 array from normalize_depth(); "
                f"got dtype={depth_uint8.dtype}. Call normalize_depth() first."
            )
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    def save_depth(self, depth_uint8_or_bgr, path):
        cv2.imwrite(path, depth_uint8_or_bgr)


DEFAULT_MODEL_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\models\midas\midas_v21_small_256.onnx"
DEFAULT_CAMERA_INDEX = 0

if __name__ == "__main__":
    import sys
    import time
    import os

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAMERA_INDEX

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"MiDaS model not found at: {model_path}\n"
            "Either fix DEFAULT_MODEL_PATH at the top of this file, or pass the "
            "correct path as the first command-line argument."
        )

    model = DepthEstimator(model_path)


    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check that the webcam is connected and not in use by another app."
        )

    print("Live depth estimation running. Press 'q' or ESC to quit.")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed — camera disconnected?")
                break

            t0 = time.time()
            depth = model.predict(frame)
            infer_ms = (time.time() - t0) * 1000

            normalized = model.normalize_depth(depth)
            colorized = model.colorize_depth(normalized)

            # Smoothed FPS over the full loop (grab + inference + display)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            combined = np.hstack([frame, colorized])

            label = f"FPS: {fps:.1f}  |  MiDaS inference: {infer_ms:.1f} ms"
            cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.imshow("Live Depth Estimation (frame | depth) — q/ESC to quit", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()