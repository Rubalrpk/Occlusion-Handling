import cv2
import numpy as np
import onnxruntime as ort


class GANRefiner:
    """ONNX Runtime wrapper for the trained boundary refinement generator.

    Takes an RGB patch + coarse depth patch (as produced by
    region_selector.py's extract_patches()) and returns a refined depth
    patch of the SAME size as the input box — resizing to/from the
    network's fixed training patch size internally, so callers never need
    to think about PATCH_SIZE.

    IMPORTANT: normalization here must exactly match train_gan.py's
    normalize_patch() — the network was trained on per-patch min-max
    normalized [0,1] depth, and its output needs to be un-normalized back
    to the original depth patch's value range before being merged back
    into the full depth map.
    """

    def __init__(self, model_path, patch_size=64):
        self.patch_size = patch_size

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("Boundary refinement GAN loaded successfully.")

    @staticmethod
    def _normalize(patch):
        p_min, p_max = float(patch.min()), float(patch.max())
        denom = p_max - p_min
        if denom < 1e-6:
            return np.zeros_like(patch, dtype=np.float32), p_min, p_max
        return (patch - p_min) / denom, p_min, p_max

    @staticmethod
    def _denormalize(patch, p_min, p_max):
        return patch * (p_max - p_min) + p_min

    def refine_patch(self, rgb_patch, depth_patch):
        """Refine a single (rgb_patch, depth_patch) pair. Returns a depth
        patch the same (H, W) as the input depth_patch — safe to hand
        straight to region_selector.merge_patches().
        """
        orig_h, orig_w = depth_patch.shape[:2]

        rgb_resized = cv2.resize(rgb_patch, (self.patch_size, self.patch_size))
        rgb_rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
        rgb_norm = rgb_rgb.astype(np.float32) / 255.0

        depth_resized = cv2.resize(depth_patch, (self.patch_size, self.patch_size),
                                    interpolation=cv2.INTER_LINEAR)
        depth_norm, p_min, p_max = self._normalize(depth_resized)

        rgb_chw = np.transpose(rgb_norm, (2, 0, 1))
        depth_chw = depth_norm[np.newaxis, :, :]
        input_tensor = np.concatenate([rgb_chw, depth_chw], axis=0)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

        raw_output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]

        refined_norm = np.squeeze(raw_output).astype(np.float32)
        refined = self._denormalize(refined_norm, p_min, p_max)

        # Resize back to the original patch's (H, W) so it matches the
        # box region_selector expects to paste it into.
        refined_full = cv2.resize(refined, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return refined_full

    def refine_all(self, patches):
        """Convenience method: takes the list of patch dicts produced by
        RegionSelector.extract_patches() and returns a new list with each
        "depth" entry replaced by the GAN-refined version — ready to pass
        straight to RegionSelector.merge_patches().
        """
        refined_patches = []
        for p in patches:
            refined_depth = self.refine_patch(p["rgb"], p["depth"])
            refined_patches.append({
                "box": p["box"],
                "rgb": p["rgb"],
                "depth": refined_depth,
            })
        return refined_patches


DEFAULT_MODEL_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\models\gan\boundary_refinement_gan.onnx"
DEFAULT_MIDAS_PATH = r"C:\Users\rubal\OneDrive\Desktop\urf\AGBR-NET\models\midas\midas_v21_small_256.onnx"
DEFAULT_CAMERA_INDEX = 0

if __name__ == "__main__":
    import sys
    import time
    import os
    from depth_estimation import DepthEstimator
    from confidence_estimator import ConfidenceEstimator
    from region_selector import RegionSelector

    gan_model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    midas_model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MIDAS_PATH
    camera_index = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CAMERA_INDEX

    if not os.path.isfile(midas_model_path):
        raise FileNotFoundError(f"MiDaS model not found at: {midas_model_path}")

    depth_model = DepthEstimator(midas_model_path)
    confidence_model = ConfidenceEstimator(depth_model)
    selector = RegionSelector()

    gan = None
    if os.path.isfile(gan_model_path):
        gan = GANRefiner(gan_model_path)
    else:
        print(f"NOTE: no trained GAN model found at {gan_model_path}")
        print("Running the pipeline WITHOUT refinement (passthrough) so you can still "
              "verify the rest of the chain works end-to-end. Train the GAN with "
              "train_gan.py and point DEFAULT_MODEL_PATH at the exported .onnx to enable it.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("Live pipeline running. Press 'q' or ESC to quit.")

    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed — camera disconnected?")
                break

            t0 = time.time()
            confidence, depth = confidence_model.estimate(frame)
            t1 = time.time()

            boxes = selector.select_regions(confidence)
            t2 = time.time()

            if gan is not None and boxes:
                patches = selector.extract_patches(frame, depth, boxes)
                t3 = time.time()
                refined_patches = gan.refine_all(patches)
                t4 = time.time()
                final_depth = selector.merge_patches(depth, refined_patches)
                t5 = time.time()
            else:
                final_depth = depth
                t3 = t4 = t5 = t2

            infer_ms = (t5 - t0) * 1000

            frame_count += 1
            if frame_count % 10 == 0:
                print(
                    f"[stage timing] confidence: {(t1-t0)*1000:.1f}ms | "
                    f"region_select: {(t2-t1)*1000:.1f}ms | "
                    f"extract: {(t3-t2)*1000:.1f}ms | "
                    f"GAN ({len(boxes)} regions): {(t4-t3)*1000:.1f}ms | "
                    f"merge: {(t5-t4)*1000:.1f}ms | "
                    f"TOTAL: {infer_ms:.1f}ms"
                )

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            normalized = depth_model.normalize_depth(final_depth)
            colorized = depth_model.colorize_depth(normalized)
            vis = selector.visualize(colorized, boxes)

            label = f"FPS: {fps:.1f}  |  Total: {infer_ms:.1f} ms  |  Regions: {len(boxes)}"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            display_width = 640
            h, w = frame.shape[:2]
            scale = display_width / w
            frame_disp = cv2.resize(frame, (display_width, int(h * scale)))
            vis_disp = cv2.resize(vis, (display_width, int(h * scale)))
            combined = np.hstack([frame_disp, vis_disp])

            cv2.imshow("Full Pipeline (frame | refined depth) - q/ESC to quit", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()