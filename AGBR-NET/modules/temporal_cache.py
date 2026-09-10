import cv2
import numpy as np


class TemporalCache:
    """Tracks state across frames so the pipeline can skip expensive
    re-processing on regions that haven't meaningfully changed since the
    last frame.

    This is the core mechanism behind the project's efficiency claim:
    without it, every frame re-runs full GAN refinement on every selected
    region regardless of whether anything in the scene actually moved.
    With it, most frames in a relatively static scene should be able to
    reuse most of the previous frame's refined output.
    """

    def __init__(self, change_threshold=15, force_refresh_interval=30):
        """
        Args:
            change_threshold: pixel intensity difference (0-255) above
                which a pixel is considered "changed" between frames.
                Lower = more sensitive to small movement/noise.
            force_refresh_interval: force a full refresh (ignore the
                cache, treat everything as changed) every N frames,
                regardless of detected change. This is a safety valve
                against silent drift — if a region is never re-refined
                because it never registers as "changed" (e.g. very slow
                lighting drift under the change_threshold), small errors
                could otherwise accumulate unnoticed indefinitely.
        """
        self.change_threshold = change_threshold
        self.force_refresh_interval = force_refresh_interval

        self.prev_frame_gray = None
        self.prev_refined_depth = None
        self.frame_index = 0

    def reset(self):
        """Call this if the camera view changes drastically (e.g. scene
        cut, camera repositioned) — otherwise the cache will compare
        against a now-irrelevant previous frame."""
        self.prev_frame_gray = None
        self.prev_refined_depth = None
        self.frame_index = 0

    def is_force_refresh_frame(self):
        """True on the very first frame (nothing cached yet) and every
        force_refresh_interval-th frame thereafter."""
        return self.frame_index == 0 or (self.frame_index % self.force_refresh_interval) == 0

    def compute_change_mask(self, frame):
        """Returns a binary mask (1 = changed, 0 = unchanged) comparing
        this frame to the last one seen. On the first frame, or after a
        reset, everything is marked changed (nothing to compare against).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame_gray is None or self.prev_frame_gray.shape != gray.shape:
            changed = np.ones(gray.shape, dtype=np.uint8)
        else:
            diff = cv2.absdiff(gray, self.prev_frame_gray)
            _, changed = cv2.threshold(diff, self.change_threshold, 1, cv2.THRESH_BINARY)

        self.prev_frame_gray = gray
        return changed

    def has_cached_depth(self, frame_shape):
        """Whether there's a usable cached full-frame depth map matching
        the given (H, W) — False on the first frame, after a reset, or
        if the frame size changed (e.g. camera resolution changed)."""
        return (
            self.prev_refined_depth is not None
            and self.prev_refined_depth.shape[:2] == frame_shape[:2]
        )

    def get_cached_patch(self, box):
        """Extract the cached depth patch for a given (x0, y0, x1, y1)
        box from the previous frame's full refined depth map. Caller
        must check has_cached_depth() first."""
        x0, y0, x1, y1 = box
        return self.prev_refined_depth[y0:y1, x0:x1].copy()

    def update(self, full_refined_depth):
        """Store this frame's final (post-merge) depth map as the cache
        for next frame, and advance the frame counter. Call this once,
        at the end of processing each frame."""
        self.prev_refined_depth = full_refined_depth.copy()
        self.frame_index += 1