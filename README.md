# Pneumonia Detection from Chest X-rays — PyTorch, Transfer Learning (ResNet-18)

**Goal:** Reproducible transfer learning pipeline on the Kaggle *Chest X-Ray Images (Pneumonia)* dataset with class-imbalance handling, augmentation, **metrics** (F1, PR-AUC), **curves** (ROC/PR), **confusion matrix**, and **Grad-CAM** sanity checks.

> **Note:** Metrics below are **provisional (typical)** for this setup. Replace with your measured results after running.

## Dataset
- Kaggle: *Chest X-Ray Images (Pneumonia)* (Kermany et al.)
- Folder structure under `data/`:
```
data/chest_xray/
├─ train/ NORMAL, PNEUMONIA
├─ val/   NORMAL, PNEUMONIA
└─ test/  NORMAL, PNEUMONIA
```

## Environment
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
printf "data/\noutputs/\n.venv/\n__pycache__/\n" > .gitignore
```

## Quickstart
```bash
python src/train.py --data_root data/chest_xray --epochs 8 --batch_size 32 --lr 3e-4 --seed 42
python src/eval.py --data_root data/chest_xray --ckpt outputs/best.pt --seed 42
python src/gradcam.py --data_root data/chest_xray --ckpt outputs/best.pt --n 6
```

## Results (Provisional)
| Model                        | F1 (test) | PR-AUC | ROC-AUC |
|----------------------------:|:---------:|:------:|:------:|
| **ResNet-18 (finetuned)**   | **0.81–0.86 → 0.83** | **0.82–0.88 → 0.84** | **0.90–0.95 → 0.93** |

Generated:
- PR/ROC: `outputs/pr_curve_test.png`, `outputs/roc_curve_test.png`
- Confusion: `outputs/confusion_matrix_test.png`
- Metrics: `outputs/metrics_test.json`
- Checkpoint: `outputs/best.pt`
- Grad-CAM: `outputs/gradcam/*.png`
