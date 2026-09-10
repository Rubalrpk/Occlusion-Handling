"""
Scans an already-downloaded dataset folder (rgb/ + depth/) for corrupted
or unreadable image files, so you know upfront how many bad files you're
dealing with instead of hitting them one at a time mid-training.

Usage:
    python check_data.py --data_dir ./data
    python check_data.py --data_dir ./data --delete_corrupted
"""

import os
import glob
import argparse
import cv2


def check_dataset(data_dir, delete_corrupted=False):
    rgb_paths = sorted(glob.glob(os.path.join(data_dir, "rgb", "*")))
    depth_dir = os.path.join(data_dir, "depth")

    print(f"Checking {len(rgb_paths)} RGB/depth pairs in {data_dir} ...")

    bad_pairs = []

    for i, rgb_path in enumerate(rgb_paths):
        depth_path = os.path.join(depth_dir, os.path.basename(rgb_path))

        rgb_ok, depth_ok = True, True

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            rgb_ok = False

        if not os.path.isfile(depth_path):
            depth_ok = False
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                depth_ok = False

        if not rgb_ok or not depth_ok:
            reason = []
            if not rgb_ok:
                reason.append("RGB unreadable")
            if not depth_ok:
                reason.append("depth missing/unreadable")
            bad_pairs.append((rgb_path, depth_path, ", ".join(reason)))

        if (i + 1) % 100 == 0:
            print(f"  checked {i + 1}/{len(rgb_paths)}...")

    print(f"\nDone. {len(bad_pairs)} bad pair(s) out of {len(rgb_paths)} total.")

    if bad_pairs:
        print("\nCorrupted/unreadable files:")
        for rgb_path, depth_path, reason in bad_pairs:
            print(f"  {os.path.basename(rgb_path)}: {reason}")

        if delete_corrupted:
            print("\nDeleting corrupted files...")
            for rgb_path, depth_path, _ in bad_pairs:
                if os.path.isfile(rgb_path):
                    os.remove(rgb_path)
                if os.path.isfile(depth_path):
                    os.remove(depth_path)
            print(f"Deleted {len(bad_pairs)} bad pair(s).")
        else:
            print("\nRun again with --delete_corrupted to remove these automatically, "
                  "or re-run download_nyu_data.py to fetch the dataset fresh if a lot "
                  "of files are bad (suggests an interrupted download).")
    else:
        print("Dataset looks clean — safe to proceed with training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--delete_corrupted", action="store_true",
                         help="Remove corrupted pairs automatically instead of just reporting them.")
    args = parser.parse_args()

    check_dataset(args.data_dir, args.delete_corrupted)