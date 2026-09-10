# ABGR-Net

**Adaptive Boundary-Guided Refinement for Confidence-Aware Monocular Occlusion Handling in Augmented Reality**

A hardware-independent occlusion handling pipeline for AR/VR that combines monocular depth estimation, semantic segmentation, confidence-aware boundary refinement, and adaptive region selection — running entirely on a standard RGB webcam with no LiDAR, stereo cameras, or depth sensors required.

> **Status: research prototype, active development.** This is an ongoing capstone research project. Numbers, architecture choices, and results below reflect the current state and will change as the project progresses toward its written paper.

---

## Pipeline

```
RGB Frame
   │
   ▼
MiDaS (monocular depth estimation)
   │
   ▼
Depth Map
   │
   ▼
Adaptive Confidence Estimation
   │
   ▼
Adaptive Region Selection
   │
   ▼
Boundary Refinement GAN   (only applied to low-confidence regions)
   │
   ▼
Refined Depth
   │
   ▼
Occlusion-Aware Rendering
```

Each stage is implemented as a standalone, independently testable module — see `modules/` below. The core idea is that boundary refinement (the most expensive step) is only applied where a lightweight confidence signal flags it as needed, rather than across the full frame every frame.

---

## Project structure

```
AGBR-NET/
├── modules/
│   ├── depth_estimation.py       # MiDaS ONNX wrapper
│   ├── segmentation.py           # U²-Net ONNX wrapper
│   ├── confidence_estimator.py   # Per-pixel confidence map from depth
│   ├── region_selector.py        # Confidence -> boxes sent to GAN refinement
│   ├── gan_architecture.py       # Generator + discriminator (PyTorch)
│   ├── train_gan.py              # Offline training script (run on Colab/GPU)
│   ├── export_model.py           # Re-export a trained checkpoint to single-file ONNX
│   ├── gan_refinement.py         # Live inference wrapper + full pipeline demo
│   ├── check_data.py             # Scans training data for corrupted files
│   └── download_nyu_data.py      # Fetches NYU Depth V2 training data
├── main.py             # TODO: describe entry point
├── config.py           # TODO: describe configuration
├── models/             # NOT in git — trained .onnx files go here (see Setup)
├── datasets/           # NOT in git — training data goes here (see Setup)
├── outputs/            # NOT in git — generated outputs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Rubalrpk/Occlusion-Handling.git
cd Occlusion-Handling
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Models you need (not included in this repo — see `.gitignore`)

| Model | Where it goes | How to get it |
|---|---|---|
| MiDaS (`midas_v21_small_256.onnx`) | `models/midas/` | Pre-trained, from the official MiDaS repo |
| U²-Net (`u2net.onnx`) | `models/u2net/` | Pre-trained, from the official U²-Net repo |
| Boundary Refinement GAN (`boundary_refinement_gan.onnx`) | `models/gan/` | Trained from scratch by this project — see Training below |

Update the `DEFAULT_MODEL_PATH` constants at the top of each module if your paths differ from the defaults.

---

## Running the pipeline

Each stage can be run standalone against your webcam, for testing and debugging:

```bash
cd modules
python depth_estimation.py          # MiDaS only
python segmentation.py              # U²-Net only
python confidence_estimator.py      # Depth + confidence map
python region_selector.py           # + adaptive region selection (shows selected boxes)
python gan_refinement.py            # Full pipeline including GAN refinement
```

All scripts default to camera index 0 and open a live window showing intermediate outputs plus an FPS/timing readout. Press `q` or `Esc` to quit.

---

## Training the boundary refinement GAN

The GAN has no pre-trained weights available anywhere — it's trained from scratch on synthetic coarse/sharp depth pairs. **Run this on Colab or another GPU machine, not locally.**

```bash
pip install datasets pillow numpy torch opencv-python onnxscript

python download_nyu_data.py --output_dir ./datasets --limit 400
python check_data.py --data_dir ./datasets          # verify no corrupted files first
python train_gan.py --data_dir ./datasets --epochs 30 --batch_size 16
```

This exports `boundary_refinement_gan.onnx` at the end — move it to `models/gan/` locally to enable GAN refinement in `gan_refinement.py`.

If you already have a `.pt` checkpoint and just need to (re-)export it as a single-file ONNX model:
```bash
python export_model.py --checkpoint ./gan_checkpoints/generator_epoch30.pt
```

---

## Current findings / known limitations

- **U²-Net is the primary throughput bottleneck** (~600ms/frame on CPU) when run full-frame; MiDaS alone is fast (~30-45ms/frame).
- **Confidence estimation and region selection are lightweight** individually, but running multiple ONNX Runtime sessions in one process (e.g. MiDaS + the GAN together) can cause CPU thread contention — session thread pools are explicitly bounded (`intra_op_num_threads`) to mitigate this.
- **Sustained inference load can trigger thermal throttling** on laptop-class hardware over a few minutes of continuous use, which is itself part of the motivation for adaptive/selective processing rather than full-frame refinement every frame.
- U²-Net (trained for general salient object segmentation, not people specifically) struggles with transparent/thin structures — glasses lenses in particular are a consistent failure case, motivating the boundary refinement stage.

## Roadmap

- [ ] Temporal reuse: cache confidence/refinement results across frames, only re-process regions that actually changed
- [ ] Quantitative evaluation against ground-truth occlusion masks (IoU, boundary accuracy)
- [ ] Compute/quality Pareto curve across confidence thresholds
- [ ] Ablation study (full pipeline vs. no-GAN vs. no-adaptive-selection vs. depth-only vs. segmentation-only)

---

## License

*(add your chosen license here — MIT is a common default for academic/research code if you don't have a specific requirement)*

## Acknowledgments

Built on [MiDaS](https://github.com/isl-org/MiDaS) (Ranftl et al.) and [U²-Net](https://github.com/xuebinqin/U-2-Net) (Qin et al.). Training data from [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) (Silberman et al.), via the [jagennath-hari/nyuv2](https://huggingface.co/datasets/jagennath-hari/nyuv2) Hugging Face mirror.
