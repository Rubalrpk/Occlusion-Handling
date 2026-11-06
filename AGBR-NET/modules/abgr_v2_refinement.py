"""ONNX inference wrapper for the five-channel ABGR-Net V2 refiner.

Input channels: RGB (0..1), locally normalized coarse MiDaS depth (0..1),
and gradient-derived confidence (0..1). Each selected region is processed
as a 64x64 patch. This module does not select regions or merge patches.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class ABGRV2Refiner:
    def __init__(self, model_path, patch_size=64, providers=None):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ABGR-Net V2 ONNX model not found: {model_path}")

        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )

        model_input = self.session.get_inputs()[0]
        model_output = self.session.get_outputs()[0]
        self.input_name = model_input.name
        self.output_name = model_output.name
        self.input_shape = model_input.shape
        self.output_shape = model_output.shape

        if len(self.input_shape) != 4:
            raise ValueError(f"Expected NCHW model input, got {self.input_shape}")
        channels = self.input_shape[1]
        if isinstance(channels, int) and channels != 5:
            raise ValueError(
                f"Expected five input channels (RGB + depth + confidence), got {channels}"
            )

        print("ABGR-Net V2 refiner loaded.")
        print(f"ONNX input: {self.input_shape}; output: {self.output_shape}")

    @staticmethod
    def normalize_depth(depth):
        """Normalize a patch using its 1st and 99th percentile values."""
        depth = np.asarray(depth, dtype=np.float32)
        d_min = float(np.percentile(depth, 1))
        d_max = float(np.percentile(depth, 99))
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth, dtype=np.float32), d_min, d_max
        normalized = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
        return normalized.astype(np.float32), d_min, d_max

    @staticmethod
    def estimate_confidence(depth):
        """Gradient-derived heuristic confidence; this is not calibrated uncertainty."""
        depth = np.asarray(depth, dtype=np.float32)
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx * gx + gy * gy)
        gradient /= float(gradient.max()) + 1e-6
        return np.clip(1.0 - gradient, 0.0, 1.0).astype(np.float32)

    def refine_patch(self, rgb_patch, depth_patch):
        if rgb_patch is None or depth_patch is None or rgb_patch.size == 0 or depth_patch.size == 0:
            raise ValueError("RGB and depth patches must be non-empty")
        if rgb_patch.ndim != 3 or rgb_patch.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 BGR frame patch, got {rgb_patch.shape}")
        if depth_patch.ndim != 2:
            raise ValueError(f"Expected a 2D depth patch, got {depth_patch.shape}")
        if rgb_patch.shape[:2] != depth_patch.shape[:2]:
            raise ValueError("RGB and depth patches must have matching spatial dimensions")

        original_h, original_w = depth_patch.shape
        size = (self.patch_size, self.patch_size)

        # Resize both modalities to the fixed training patch dimensions.
        rgb = cv2.resize(rgb_patch, size, interpolation=cv2.INTER_LINEAR)
        coarse = cv2.resize(
            depth_patch.astype(np.float32), size, interpolation=cv2.INTER_LINEAR
        )

        # Training input order: RGB, normalized coarse depth, confidence.
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        coarse_norm, d_min, d_max = self.normalize_depth(coarse)
        confidence = self.estimate_confidence(coarse)

        tensor = np.concatenate(
            [
                np.transpose(rgb, (2, 0, 1)),
                coarse_norm[None, :, :],
                confidence[None, :, :],
            ],
            axis=0,
        )[None, ...].astype(np.float32)

        prediction = self.session.run(
            [self.output_name], {self.input_name: tensor}
        )[0]
        prediction = np.asarray(prediction, dtype=np.float32)

        if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[1] != 1:
            raise RuntimeError(
                f"Expected model output (1, 1, H, W), got {prediction.shape}"
            )
        refined_norm = prediction[0, 0]
        if not np.isfinite(refined_norm).all():
            raise RuntimeError("The ONNX model returned NaN or infinite values")

        refined = refined_norm * (d_max - d_min) + d_min
        return cv2.resize(
            refined.astype(np.float32),
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

    def refine_patches(self, frame, depth, boxes):
        """Return patch dictionaries compatible with RegionSelector.merge_patches()."""
        if frame.shape[:2] != depth.shape[:2]:
            raise ValueError("Frame and depth map must have matching dimensions")

        results = []
        height, width = depth.shape[:2]
        for box in boxes:
            x0, y0, x1, y1 = map(int, box)
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"Invalid patch box {box} for frame size {(width, height)}")

            refined = self.refine_patch(
                frame[y0:y1, x0:x1],
                depth[y0:y1, x0:x1],
            )
            results.append({"box": (x0, y0, x1, y1), "depth": refined})
        return results
