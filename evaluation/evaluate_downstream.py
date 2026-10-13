"""Does synthetic data improve REAL-world disease classification?  (the question that matters)

    python -m evaluation.evaluate_downstream --real_dir data/processed --synthetic_dir outputs/filtered \
        --ratios 0 0.1 0.25 0.5 1.0 --seeds 0 1 2 --arch resnet50

* Training set  : real train (+ a fraction of synthetic images of the same classes)
* Model selection: REAL validation split (macro-F1) - early stopping
* Test          : REAL test split ONLY, never touched during training/selection
Results: downstream_results.csv (every run), downstream_summary.csv/.md (mean +- std over seeds), per-run
confusion matrices and per-class reports.  Several seeds are used because the effect of synthetic data is often
smaller than seed-to-seed variance; do not read small differences as evidence.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common.config import load_config
from common.errors import UserError, run_cli
from common.gpu import get_device
from common.imageio_utils import list_images
from common.logging_utils import setup_logging
from common.seed import seed_everything
from evaluation.visualization import confusion_plot
from preprocessing.build_dataset import check_synthetic_provenance, select_synthetic

log = logging.getLogger("evaluate_downstream")
ARCHS = ["resnet50", "efficientnet_b0", "convnext_tiny", "mobilenet_v3_large"]


class ImgDS(Dataset):
    def __init__(self, paths: List[str], labels: List[int], tf):
        self.p, self.y, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(self.p[i]) as im:
            return self.tf(im.convert("RGB")), self.y[i]


def build_classifier(arch: str, n_classes: int, pretrained: bool) -> nn.Module:
    import torchvision.models as tvm
    if arch not in ARCHS:
        raise UserError(f"Unknown arch '{arch}'. Choose from {ARCHS}.")
    w = "DEFAULT" if pretrained else None
    try:
        m = getattr(tvm, arch)(weights=w)
    except Exception as e:
        raise UserError(f"Could not load pretrained weights for {arch} ({e}). Check your connection or use --no_pretrained.") from e
    if arch == "resnet50":
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch in ("efficientnet_b0", "mobilenet_v3_large"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def _transforms(size: int):
    import torchvision.transforms as T
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                       T.ColorJitter(0.15, 0.15, 0.1), T.ToTensor(), norm])
    ev = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), norm])
    return train, ev


def load_split_dir(real_dir: Path, split: str, c2i: Dict[str, int]):
    paths, labels = [], []
    for c, i in c2i.items():
        for p in list_images(real_dir / split / c):
            paths.append(str(p.resolve())); labels.append(i)
    return paths, labels


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu().numpy()); ys.append(y.numpy())
    return (np.concatenate(ys), np.concatenate(ps)) if ys else (np.array([]), np.array([]))


def train_eval(train_p, train_y, val, test, n_classes, cfg: Dict[str, Any], seed: int, device, workers: int):
    seed_everything(seed)
    ttf, etf = _transforms(cfg["image_size"])
    mk = lambda p, y, tf, sh: DataLoader(ImgDS(p, y, tf), cfg["batch_size"], shuffle=sh, num_workers=workers, drop_last=sh and len(p) > cfg["batch_size"])
    tl, vl, te = mk(train_p, train_y, ttf, True), mk(*val, etf, False), mk(*test, etf, False)
    model = build_classifier(cfg["arch"], n_classes, cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"] * max(len(tl), 1))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = nn.functional.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        yv, pv = predict(model, vl, device)
        f1 = precision_recall_fscore_support(yv, pv, average="macro", zero_division=0)[2]  # REAL val macro-F1
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return best_f1, *predict(model, te, device)


def test_metrics(y, p, classes: Sequence[str]) -> Dict[str, Any]:
    labels = list(range(len(classes)))
    pm, rm, fm, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y, p), "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
            "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
            "confusion_matrix": confusion_matrix(y, p, labels=labels),
            "per_class": classification_report(y, p, labels=labels, target_names=list(classes), output_dict=True, zero_division=0)}


def evaluate_downstream(real_dir: str, synthetic_dir: Optional[str], output_dir: str, dcfg: Dict[str, Any], device_name: str = "auto",
                        workers: int = 2, max_train_per_class: Optional[int] = None) -> pd.DataFrame:
    rdir, out = Path(real_dir), Path(output_dir)
    classes = sorted(p.name for p in (rdir / "train").iterdir() if p.is_dir()) if (rdir / "train").exists() else []
    if not classes:
        raise UserError(f"{rdir}/train/<class>/ not found. --real_dir must contain train/val/test class folders "
                        "(the output of preprocessing.prepare_dataset).")
    c2i = {c: i for i, c in enumerate(classes)}
    tr_p, tr_y = load_split_dir(rdir, "train", c2i)
    va = load_split_dir(rdir, "val", c2i)
    te = load_split_dir(rdir, "test", c2i)
    if not va[0] or not te[0]:
        raise UserError("Real val and test splits are required for downstream evaluation.")
    syn = None
    if synthetic_dir:
        sdir = Path(synthetic_dir)
        syn = pd.read_csv(sdir / "filtered_metadata.csv")
        syn = syn[syn["class"].isin(classes)]
        check_synthetic_provenance(syn)
        syn["abs"] = [str((sdir / p).resolve()) for p in syn["path"]]
    # --- leakage guard: test images must be real & disjoint from every training image ------------------------
    test_set = set(te[0])
    if test_set & set(tr_p) or test_set & set(va[0]) or (syn is not None and test_set & set(syn["abs"])):
        raise RuntimeError("Data leakage: a test image also appears in train/val/synthetic.")
    counts = pd.Series(tr_y).map({v: k for k, v in c2i.items()}).value_counts().to_dict()
    dev = get_device(device_name)
    rows, out_runs = [], out / "downstream_runs"
    out_runs.mkdir(parents=True, exist_ok=True)
    for ratio in dcfg["ratios"]:
        for seed in dcfg["seeds"]:
            name = f"{'real_only' if ratio == 0 else f'syn_{ratio:g}'}_seed{seed}"
            p, y = list(tr_p), list(tr_y)
            n_syn = 0
            if ratio > 0:
                if syn is None:
                    raise UserError("--synthetic_dir is required for ratios > 0.")
                sel = select_synthetic(counts, syn, ratio, dcfg["ratio_mode"], seed)
                p += list(sel["abs"]); y += [c2i[c] for c in sel["class"]]; n_syn = len(sel)
            if max_train_per_class:
                keep = []
                for c in range(len(classes)):
                    idx = [i for i, yy in enumerate(y) if yy == c][:max_train_per_class]
                    keep += idx
                p, y = [p[i] for i in keep], [y[i] for i in keep]
            t0 = time.time()
            best_val, yt, pt = train_eval(p, y, va, (te[0], te[1]), len(classes), dcfg, seed, dev, workers)
            m = test_metrics(yt, pt, classes)
            log.info("%s | real=%d syn=%d | val_macroF1=%.4f | TEST acc=%.4f macroF1=%.4f (%.0fs)", name, len(p) - n_syn, n_syn,
                     best_val, m["accuracy"], m["f1_macro"], time.time() - t0)
            confusion_plot(m["confusion_matrix"], classes, out_runs / f"confusion_{name}.png", name)
            pd.DataFrame(m["per_class"]).T.to_csv(out_runs / f"per_class_{name}.csv")
            rows.append({"ratio": ratio, "seed": seed, "n_train_real": len(p) - n_syn, "n_train_synthetic": n_syn, "val_f1_macro": best_val,
                         **{k: v for k, v in m.items() if k not in ("confusion_matrix", "per_class")}})
            pd.DataFrame(rows).to_csv(out / "downstream_results.csv", index=False)
    df = pd.DataFrame(rows)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    summ = df.groupby("ratio")[metrics].agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(out / "downstream_summary.csv")
    with open(out / "downstream_summary.md", "w") as f:
        f.write(f"# Downstream results (arch={dcfg['arch']}, pretrained={dcfg['pretrained']}, seeds={list(dcfg['seeds'])})\n\n")
        f.write("Evaluated on the REAL test split only. mean/std over seeds.\n\n" + summ.to_markdown() + "\n\n")
        f.write("Interpretation: a difference smaller than the seed std is not evidence of improvement.\n")
    log.info("Summary:\n%s", summ.to_string())
    return df


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", required=True, help="processed dir with train/val/test/<class>/")
    ap.add_argument("--synthetic_dir", default=None, help="quality-controlled synthetic dir (filtered_metadata.csv)")
    ap.add_argument("--output_dir", default="./outputs/metrics/downstream")
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--arch", choices=ARCHS, default=None)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--ratios", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--image_size", type=int, default=None)
    ap.add_argument("--ratio_mode", choices=["per_class", "balance_to_max"], default=None)
    ap.add_argument("--max_train_per_class", type=int, default=None, help="quick tests only")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    setup_logging(Path(a.output_dir) / "evaluate_downstream.log")
    d = load_config(a.config, a.set)["downstream"]
    for k in ("arch", "ratios", "seeds", "epochs", "batch_size", "image_size", "ratio_mode"):
        if getattr(a, k) is not None:
            d[k] = getattr(a, k)
    if a.no_pretrained:
        d["pretrained"] = False
    evaluate_downstream(a.real_dir, a.synthetic_dir, a.output_dir, d, a.device, a.num_workers, a.max_train_per_class)


if __name__ == "__main__":
    run_cli(main)
