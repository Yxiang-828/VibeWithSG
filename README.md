# Team SyntaxError — Offroad Autonomy Semantic Segmentation

**Team name:** SyntaxError  
**Team members:** Yao Xiang (solo)  
**Challenge:** Duality AI — Offroad Autonomy Semantic Segmentation (desert biome)  
**Final test mIoU:** **0.4214** on 1002 `testImages` (pixel accuracy 0.691)

Submission for Duality AI's Offroad Autonomy Semantic Segmentation challenge.

### 🎯 Submission Highlights
- **Final Result:** 0.4214 mIoU (Test) / 0.6361 mIoU (Validation)
- **Core Deliverables:** `HACKATHON_REPORT.pdf`, `test_metrics.json`, `predictions/`, `weights/`
- **Reproduction:** See §2 for bit-accurate reproduction instructions.

## 0. Headline numbers

| Metric | Value |
|---|---|
| **mIoU (OOD Test / In-Dist Val)** | **0.4214** / **0.6361** |
| Test pixel accuracy | 0.691 |
| Model | DINOv2-L + DPT (unfrozen backbone with LLRD) |
| Checkpoint | `weights/best.pt` (2.0 GB, as 23 chunks) |

See `HACKATHON_REPORT.pdf` §4 for an honest explanation of why val and test differ.

---

## 1. Step-by-step instructions to run and test the model

### 1.1 Reconstruct the model weights (GitHub clones only)

The checkpoint `weights/best.pt` (~2.0 GB) is stored as 23 chunks under `weights/chunks/` because GitHub rejects files larger than 100 MB. Reassemble it before running any script:

```bash
cat weights/chunks/best.pt.part-* > weights/best.pt
sha256sum -c weights/best.pt.sha256
# expected: weights/best.pt: OK
```

Zipped submissions already include the merged `weights/best.pt`; skip this step.

### 1.2 Activate the environment

See §3 below for first-time setup.

```bash
source ~/venvs/seg/bin/activate       # Linux (setup_env.sh)
# or
conda activate EDU                    # Windows (setup_env.bat)
```

### 1.3 Run inference on the test images

The inference script is `scripts/test.py`. The submitted predictions were generated with multi-scale + horizontal-flip TTA:

```bash
python scripts/test.py \
    --checkpoints weights/best.pt \
    --image-dir <PATH>/Offroad_Segmentation_testImages/Color_Images \
    --output-dir predictions/ \
    --image-size 532 952 \
    --multiscale 420 532 644 \
    --hflip-tta \
    --save-format raw_id
```

This writes 1002 predicted segmentation masks to `predictions/`. See §4 for the expected output format.

Faster single-scale inference (~75 ms/image on H100, 0.4194 mIoU instead of 0.4214):

```bash
python scripts/test.py --checkpoints weights/best.pt \
    --image-dir .../testImages/Color_Images \
    --output-dir predictions/ --hflip-tta --save-format raw_id
```

---

## 2. How to reproduce the final results

### 2.1 Reported headline

**Test mIoU = 0.4214** on the 1002-image `testImages` split, with multi-scale (scales {420, 532, 644}) + horizontal-flip test-time augmentation.

**Validation mIoU = 0.6361** on the in-distribution training-set validation split. This number is stored inside the checkpoint and can be read without running anything:

```bash
python -c "import torch; c=torch.load('weights/best.pt', map_location='cpu', weights_only=False); print('phase=', c['config']['phase'], 'epoch=', c['epoch'], 'val_miou=', round(c['val_miou'], 4))"
# expected: phase= P4A epoch= ~79 val_miou= 0.6361
```

The authoritative test metrics (including per-class IoU and the 11×11 confusion matrix) are saved in `test_metrics.json`.

### 2.2 Retraining from scratch

Full training recipe (DINOv2-L frozen → full-resolution decoder → backbone fine-tune):

```bash
# Phase 1: half-resolution warmup, 40 epochs, decoder-only
python scripts/train.py --config configs/01_dinov2_l_dpt.yaml --run-name P1

# Phase 2: full-resolution continuation, 20 epochs, decoder-only, resume from P1 best
python scripts/train.py --config configs/02_full_res.yaml --run-name P2 \
    --resume runs/P1/best.pt

# Phase 4A: unfreeze last 8 backbone blocks with LLRD 0.75, 20 epochs
python scripts/train.py --config configs/04_unfreeze_llrd.yaml --run-name P4A \
    --resume runs/P2/best.pt
```

Reference epoch times (batch size 2, full resolution 532×952):
- NVIDIA H100 80 GB, bf16: ~60 s/epoch
- AMD RX 6800 XT 16 GB, fp32: ~6 min/epoch at half resolution only

### 2.3 Regenerating the submitted masks

Re-run §1.3 after completing §1.1 and §1.2. Outputs are bit-level non-deterministic across different GPU vendors and BLAS backends; pixel-level variation is well under 0.1% and does not materially change IoU.

---

## 3. Environment / dependency requirements

### 3.1 Hardware tested

- **Primary training:** NVIDIA H100 80 GB (CUDA 12)
- **Local development:** AMD RX 6800 XT 16 GB (ROCm 6.2)
- **CPU fallback:** supported for inference only, ~5–10 s per image single-scale

Minimum VRAM for full-resolution multi-scale inference with ViT-L: 10 GB. For single-scale: 6 GB.

### 3.2 Install

Linux / macOS:
```bash
bash ENV_SETUP/setup_env.sh
source ~/venvs/seg/bin/activate
```

Windows (per hackathon instructions §2.i.3):
```bat
ENV_SETUP\setup_env.bat
conda activate EDU
```

Any existing Python ≥ 3.10 environment:
```bash
pip install -r ENV_SETUP/requirements.txt
```

### 3.3 Key dependencies

Pinned in `ENV_SETUP/requirements.txt`. Highlights:
- `torch` ≥ 2.3
- `numpy`, `pillow`, `pyyaml`
- The DINOv2 backbone is loaded via `torch.hub` on first run and cached under `~/.cache/torch/hub/`.

---

## 4. Expected outputs and how to interpret them

### 4.1 Prediction masks

`scripts/test.py` writes one PNG per input image to `--output-dir`.

| Property | Value |
|---|---|
| File count | 1002 (one per input image in `Offroad_Segmentation_testImages/Color_Images/`) |
| Filename | matches input (e.g. `0000060.png`) |
| Image size | **960 × 540** (width × height, same as input) |
| Dtype | `uint16` |
| Pixel values | raw dataset IDs: `{0, 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}` |

Raw-ID → class mapping (from the hackathon documentation §1 Data Overview):

| ID | Class |
|---|---|
| 100 | Trees |
| 200 | Lush Bushes |
| 300 | Dry Grass |
| 500 | Dry Bushes |
| 550 | Ground Clutter |
| 600 | Flowers |
| 700 | Logs |
| 800 | Rocks |
| 7100 | Landscape |
| 10000 | Sky |
| 0 | Background (outside any of the above) |

### 4.2 Output verification

Run this after §1.3 to confirm the outputs are valid:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np
p = Path('predictions')
allowed = {0,100,200,300,500,550,600,700,800,7100,10000}
files = sorted(p.glob('*.png'))
vals, sizes = set(), set()
for f in files:
    with Image.open(f) as im:
        sizes.add(im.size)
        vals |= set(map(int, np.unique(np.asarray(im))))
assert len(files) == 1002, f"expected 1002 masks, got {len(files)}"
assert sizes == {(960, 540)}, f"bad sizes: {sizes}"
assert vals <= allowed, f"unexpected values: {vals - allowed}"
print("OK: 1002 masks, 960x540, raw IDs in range")
PY
```

Expected output: `OK: 1002 masks, 960x540, raw IDs in range`.

### 4.3 Report artifacts

- `HACKATHON_REPORT.pdf` — performance evaluation and analysis report covering methodology, training journey, per-class IoU, val/test distribution-shift explanation, TTA breakdown, and honest limitations.
- `HACKATHON_REPORT.md` — markdown source for the PDF.
- `test_metrics.json` — per-class IoU, pixel accuracy, and 11×11 confusion matrix on the 1002-image test set. Authoritative source for all numbers.
- `runs/final/phase_miou_bar.png` — validation mIoU by training phase (P0 → P4A).
- `runs/final/per_class_iou.png` — validation vs test per-class IoU (final model).
- `runs/final/tta_breakdown.png` — test mIoU gain from HFlip and multi-scale TTA.

---

## 5. Package contents

```
VibeWithSG-submission/
├── README.md                 # this file
├── HACKATHON_REPORT.pdf                # hackathon report
├── HACKATHON_REPORT.md                 # markdown source of the report
├── test_metrics.json         # authoritative test metrics + confusion matrix
├── configs/
│   ├── final.yaml            # locked config used for the submitted weights
│   └── 0*.yaml               # per-phase training configs (P0, P1, P2, P3, P4A)
├── seg/                      # model / dataset / metrics Python package
├── scripts/
│   ├── train.py              # training entry point
│   ├── test.py               # inference wrapper (required filename per hackathon spec)
│   ├── predict.py            # single/ensemble + HFlip + multi-scale TTA
│   └── precompute_masks.py
├── ENV_SETUP/
│   ├── setup_env.sh          # Linux
│   ├── setup_env.bat         # Windows
│   └── requirements.txt
├── weights/
│   ├── best.pt               # P4A checkpoint (reassemble from chunks, §1.1)
│   ├── best.pt.sha256
│   └── chunks/               # 23 × ≤90 MB parts for GitHub
├── predictions/              # 1002 submitted prediction masks (§4.1)
└── runs/final/               # figures used in HACKATHON_REPORT.pdf
```
