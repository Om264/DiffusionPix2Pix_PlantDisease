# DiffusionPix2Pix for Plant-Disease Image Synthesis

> **Research-oriented implementation inspired by the published DiffusionPix2Pix methodology.**
> This is **not** the authors' code and is not claimed to be identical to it (see [section 2](#2-what-diffusionpix2pix-is-and-what-this-repo-does-and-does-not-reproduce)).

A modular, reproducible pipeline that takes a (small) real plant-disease dataset and produces a **class-balanced,
quality-controlled, de-duplicated synthetic dataset of ≥ 10,000 images**, then answers the question that matters:
**do the synthetic images improve classification on untouched REAL test images?**

Reference paper: *"When synthetic plants get sick: Disease graded image datasets by novel regression-conditional diffusion
models"*, Computers and Electronics in Agriculture, 2025, vol. 229, article 109690, DOI
[10.1016/j.compag.2024.109690](https://doi.org/10.1016/j.compag.2024.109690).

---------------------------------------------------------------------------------------------------------------------

## Status: what has and has not been executed

| Item | Status |
|---|---|
| Unit tests (`pytest tests`, 34 tests: data, splits, leakage, masks, model, scheduler, sampling, QC, dedup, metrics, build_dataset, train + resume) | **Executed** (CPU, Python 3.12, torch 2.14, diffusers 0.41, transformers 5.18) → all passed |
| Smoke pipeline `run_pipeline.py --total_images 100 --preset smoke` on procedural **dummy** leaves | **Executed** on CPU, mechanics only – see [Smoke test](#smoke-test-what-it-proves-and-what-it-does-not) |
| Training / generation on a **GPU**, memory use, speed (RTX 3090) | **Not yet executed** |
| Any real plant-disease dataset | **Not yet executed** |
| Latent mode with a *pretrained* VAE / pretrained UNet | **Not yet executed** (only randomly-initialised stand-ins in tests) |
| DINOv2 / CLIP with real pretrained weights | **Not yet executed** (loader tested with a tiny random local DINOv2) |
| LPIPS with real AlexNet weights | **Not yet executed** (code path tested with a random backbone) |
| Inception-v3 features for FID/KID | Feature extractor executed (weights downloaded from GitHub); **no real FID/KID result exists** |
| Multi-GPU (DDP) | **Not yet executed** (helper provided, untested) |
| Accuracy / F1 / FID / KID / LPIPS / SSIM / PSNR on real data | **Not yet executed. No such numbers are reported anywhere in this repo.** |

Everything in `outputs/` is produced by *your* runs; the pipeline report (`final_report.md`) contains only values the run measured.

---------------------------------------------------------------------------------------------------------------------

## Table of contents
1. [Project motivation](#1-project-motivation)
2. [What DiffusionPix2Pix is, and what this repo does and does not reproduce](#2-what-diffusionpix2pix-is-and-what-this-repo-does-and-does-not-reproduce)
3. [Architecture](#3-architecture)
4. [Pipeline](#4-pipeline)
5. [Installation](#5-installation)
6. [Dataset preparation](#6-dataset-preparation)
7. [Mask preparation](#7-mask-preparation)
8. [Training (and exactly what training data is required)](#8-training)
9. [Generation](#9-generation)
10. [Generating ≥ 10,000 images](#10-generating--10000-images)
11. [Quality filtering](#11-quality-filtering)
12. [Deduplication](#12-deduplication)
13. [Evaluation of the generator (metrics explained)](#13-evaluation-of-the-generator)
14. [Downstream classification](#14-downstream-classification)
15. [Configuration](#15-configuration)
16. [Troubleshooting](#16-troubleshooting)
17. [GPU requirements](#17-gpu-requirements)
18. [Expected folder structure](#18-expected-folder-structure)
19. [Reproducibility](#19-reproducibility)
20. [Research limitations](#20-research-limitations)
21. [Citation](#21-citation)
22. [Extending to the thesis pipeline](#22-extending-to-the-thesis-pipeline-ssl-pseudo-labels-detection)

---------------------------------------------------------------------------------------------------------------------

## 1. Project motivation

Plant-disease datasets are small, imbalanced and rarely contain graded severity. Classifiers trained on them overfit and
fail on minority diseases. Generative models can create extra training images, but a pretty image is not necessarily a
*useful* or *biologically plausible* one. This repo therefore treats generation as a **controlled image-to-image edit**
(keep the leaf, its texture, lighting and background; change only the disease), puts **hard quality gates** after
generation, keeps **real train/val/test strictly separated**, and measures the effect on **real-world classification**.

## 2. What DiffusionPix2Pix is, and what this repo does and does not reproduce

From the paper's abstract (the only part of the paper used to write this code): DiffusionPix2Pix is a diffusion-based
**image-to-image** translation approach for plant disease synthesis that uses **image priors** of the plant, **arbitrary
binary masks**, and a **regression (graded) condition** so that disease **severity is controllable**.

| Concept in the paper | Implemented here |
|---|---|
| Diffusion image-to-image translation | Conditional DDPM/DDIM; the denoiser sees `[x_t, source image, mask]` (channel concatenation, as in Pix2Pix/InstructPix2Pix-style editing) |
| Image prior | The *source image* (a real healthy leaf at inference; a disease-suppressed version of the target during training) |
| Arbitrary binary masks | `mask` channel into the UNet + mask-weighted loss + **background compositing** at sampling time (outside the mask the source pixels are pasted back) |
| Disease conditioning | `DiseaseEmbedding(class_id)` (ids are generated from your folder names; extra *null* id for classifier-free guidance) |
| Regression-conditional / graded severity | `SeverityEmbedding(s ∈ [0,1])` = Fourier features + MLP, with a learned *null* vector for unknown severity; classifier-free guidance on (disease, severity) |

**Not known / not reproduced** (the paper's details were not available to this implementation): exact network, how training
pairs were built, loss, how severity is defined/measured, datasets, hyper-parameters. Choices made here (diffusers UNet,
cosine schedule, ε-prediction, min-SNR weighting, pseudo-prior pairing, …) are mine and are documented in the code.
No claim is made that results will match the paper's.

## 3. Architecture

```
 real leaf (source prior) ─┐
 binary mask (0 bg / 1 leaf)├───────────── channel concat ────────────┐
 noisy target x_t ─────────┘                                          ▼
                                                      ┌───────────────────────────────┐
 disease id ─► DiseaseEmbedding ─┐                    │   UNet2DConditionModel        │
 severity   ─► SeverityEmbedding ┼► tokens ──────────►│   (diffusers, cross-attention)│──► ε̂ (or v̂)
 source+mask ─► ImageEncoder ────┘     │              │                               │
                                       └► MLP ─► global vector ─► added to timestep emb.│
 diffusion step t ─► sinusoidal time embedding (inside the UNet) ─────────────────────►│
                                                      └───────────────────────────────┘
 latent mode only:  image ─► frozen VAE encoder ─► z   ...   ẑ ─► frozen VAE decoder ─► image
 sampling:  DDIM loop (+ classifier-free guidance, optional SDEdit-style strength) ─► composite(mask, source) ─► output
```

| Component | File |
|---|---|
| Image encoder, condition encoder | `models/conditioning.py` |
| Disease / severity embeddings | `models/embeddings.py` |
| Noise scheduler (q-sample, ε/v targets, DDIM step) | `models/scheduler.py` |
| Model: UNet, VAE (optional), `training_loss`, `predict` (+CFG) | `models/diffusion_pix2pix.py` |
| Mask-weighted loss, min-SNR weighting | `models/losses.py` |
| Sampling loop + compositing | `generation/sampling.py` |

Presets (`model.preset`): `tiny` (CPU smoke tests), `small`, `base`. Attention is only used at low-resolution UNet levels.
**Pixel-space** diffusion is the default (no downloads needed). **Latent** diffusion (`model.latent=true`) needs a diffusers
`AutoencoderKL` (`model.vae_path`, e.g. a local copy of `stabilityai/sd-vae-ft-mse`) and can start from a pretrained SD-style
UNet (`model.unet_pretrained`; `conv_in` is expanded to the extra channels with zero-initialised weights, InstructPix2Pix-style).
This is **image-conditioned**, not text-to-image: no text encoder is used anywhere.

## 4. Pipeline

```
REAL IMAGES (Format A or B)
  │ preprocessing.prepare_dataset  → validate, stratified split, LEAKAGE check, metadata.csv, classes.json
  ▼
processed/{train,val,test}/<class>/   (REAL only)
  │ preprocessing.prepare_masks    → leaf masks (existing | hsv | grabcut | torchscript | custom)
  │ preprocessing.estimate_severity→ severity condition (real labels kept; estimates flagged "estimated")
  ▼
training.train  (AMP, grad-accum, grad-checkpointing, EMA, resume)  → checkpoints/{latest,best}.pt
  ▼
generation.generate  (class quotas, species-matched TRAIN sources, unique seed per image) → generated/ + generation_metadata.csv
  ▼
quality.quality_control  (validity, size, blur, noise, exposure, mask consistency, disease signal, colours, source similarity)
  ▼
quality.deduplicate  (pHash + embedding cosine, per class)  → filtered/ + quality_report.csv
  ▼  (run_pipeline.py repeats generation rounds to fill per-class deficits, then trims to the balanced quotas)
preprocessing.build_dataset → combined/real_only, combined/real_plus_syn_{0.1,0.25,0.5,1}   (val/test REAL ONLY)
  ▼
evaluation.evaluate_generation (FD/FID, KID, SSIM, PSNR, LPIPS, embedding similarity, plots)
evaluation.evaluate_downstream (classifier: real-only vs real+synthetic, REAL test set only)
```

## 5. Installation

```bash
conda env create -f environment.yml && conda activate dpp-plant     # Python 3.10, CUDA 12.x wheels
# or:  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest -q tests                                                      # ~15 s on CPU
```
Tested versions are listed in the status table (Python 3.12, CPU). Python 3.10/3.11 are recommended but not yet executed here.
Note: the data package is named `dataio/` (not `datasets/`) so that it never shadows the Hugging Face `datasets` library.

## 6. Dataset preparation

Input formats (auto-detected, classes = sub-folder names; nothing is hard-coded):
```
Format A                         Format B
data/raw/<class>/*.jpg           data/raw/{train,val,test}/<class>/*.jpg   (also: valid/validation/testing)
```
```bash
python -m preprocessing.prepare_dataset --data_dir ./data/raw --output_dir ./data/processed
# optional: --severity_csv labels.csv (columns image,severity in [0,1]; --severity_scale 100 for percent)
#           --val_frac 0.1 --test_frac 0.2 --max_side 512 --min_images_per_class 5 --seed 42
```
**Strict separation / no contamination**
* Format A: stratified random split per class. Format B: your splits are kept (missing val/test are carved from train, with a warning).
* **Leakage check**: exact (MD5) and near-duplicate (pHash Hamming ≤ `--leak_hamming`, default 2) images that appear in two splits are
  removed from the lower-priority split (test > val > train). *Limitation:* rotated/flipped copies of the same physical leaf (e.g. PlantVillage
  augmentations) are not detected; if your dataset has such groups, split by group yourself (Format B).
* Synthetic images are **never** written to `processed/`. Generation sources come from the **train** split only (checked in tests), `build_dataset`
  refuses synthetic images whose `source_split != train`, and `evaluate_downstream` aborts if any test image also appears in train/val/synthetic.

Beginner-friendly errors, e.g. `No class folders were found`, `Only 2 images were found for class 'X'`, are raised for broken inputs.

## 7. Mask preparation

Mask semantics: **1 = leaf/plant, disease may be synthesised; 0 = background, must not be modified.**
```bash
python -m preprocessing.prepare_masks --processed_dir ./data/processed --method hsv          # training-free
python -m preprocessing.prepare_masks --processed_dir ./data/processed --method grabcut      # tighter, slower
python -m preprocessing.prepare_masks --processed_dir ./data/processed --method existing --masks_dir ./my_masks   # your / manual masks
python -m preprocessing.prepare_masks --processed_dir ./data/processed --method torchscript --model_path seg.pt  # learned model
python -m preprocessing.prepare_masks --processed_dir ./data/processed --segmenter_module mypkg.seg:MySegmenter  # replace the model
```
* `existing` expects `masks_dir/<class>/<image_stem>.png` (or `masks_dir/<stem>.png`; `_mask`/`_seg` suffixes accepted). Manual masks use the same layout.
* To plug in SAM/U-Net/etc.: subclass `dataio.masks.LeafSegmenter` (one method, `segment(img_rgb) -> uint8 {0,1}`) and pass `--segmenter_module`.
* Masks with coverage < 5 % or > 98 % fall back to a full-image mask (flagged in `metadata.csv: mask_source = fallback_*`); a warning appears if > 20 % fall back.
  **Look at `masks_preview.png`.** The HSV method works for fairly uniform backgrounds; cluttered field images need a learned model or your own masks.

## 8. Training

```bash
python -m training.train --config configs/training.yaml       # resumes with --resume ; override with --set train.max_steps=50000 data.resolution=256
```
Features: mixed precision (`fp16`/`bf16`/`fp32`), gradient accumulation, gradient checkpointing (`model.gradient_checkpointing=true`),
EMA weights, cosine LR with warm-up, min-SNR weighted loss, classifier-free-guidance dropout, CUDA peak-memory logging,
`latest.pt` (+ optimizer, atomic writes) / `best.pt` (best EMA validation loss), `training_history.csv`, `training.log`, SIGINT-safe checkpointing,
friendly out-of-memory message. Single GPU / CPU only; `training/distributed.py` is an optional, **untested** DDP helper.

### Training data requirements (read this)

DiffusionPix2Pix is a *paired* image-to-image model: it needs `(source image, mask, condition) → target image`.
Public plant-disease datasets almost never contain **the same leaf healthy and then diseased**, so this repo supports two pairing strategies
(`data.pairing`):

1. **`pseudo_prior` (default, no paired data needed).** For every real (usually diseased) training image `T` with leaf mask `M`:
   * edit region `E ⊆ M` – the whole leaf, or (50 %) random ellipses inside it;
   * source `S` = `T` outside `E`; inside `E` a *disease-suppressed prior* of `T` (mask-normalised Gaussian blur + partial desaturation + blend toward the mean leaf colour);
   * the model learns to restore lesion detail inside `E` given the disease class and severity.
   At generation time `S` is a **real healthy leaf** and `E` its leaf mask. **Known risks:** (i) a blurred diseased leaf still leaks large-lesion colour,
   so the model may partly "copy" instead of "create" lesions; (ii) train/inference gap (blurred prior vs. sharp healthy leaf). Both are mitigated
   (stronger blur/desaturation, classifier-free dropout, source-similarity and disease-signal QC) but **not eliminated** – inspect samples.
2. **`explicit` – real pairs.** If you have genuine healthy→diseased pairs (e.g. longitudinal images of the same plants, or pairs you built with
   an inpainting tool), list them in a CSV with columns `source_path,target_path,class[,severity,mask_path,split]` and set
   `data.pairing=explicit data.pairs_csv=pairs.csv`. This is the most faithful setting.

**Masks are required** for good behaviour (otherwise the whole image is editable and the background may be modified): see section 7.

**Severity labels.** Real graded labels (e.g. expert scores, measured lesion area) go in `--severity_csv` and are stored as
`severity_source = real`. If you have none, `estimate_severity` assigns a **colour-heuristic estimate** (fraction of non-green leaf pixels converted to a
within-class percentile) stored as `severity_source = estimated`. Estimated severity is **not ground truth**: it only provides a *conditioning signal* that is
monotone with visible discoloured area. Control with `data.severity_mode = real | estimated | none` (which provenance may condition the model); every
generated image records `severity_kind = "synthetic_condition (NOT a measured severity)"`.

## 9. Generation

```bash
python -m generation.generate --checkpoint ./outputs/checkpoints/best --total_images 500 --output_dir ./outputs/generated
python inference.py --checkpoint outputs/checkpoints/best.pt --image leaf.jpg --class Tomato___Late_blight --severity 0.6 --num_samples 4
python -m evaluation.visualization severity --checkpoint outputs/checkpoints/best.pt --class Tomato___Late_blight --severities 0.2 0.4 0.6 0.8 1.0
python -m evaluation.visualization grid --synthetic_dir outputs/filtered      # source | mask | synthetic
```
* **Sources**: real healthy leaves of the **same species** (`species = text before species_sep`, default `___`) from the **train** split (`source_pool=auto`). If a species has
  no healthy images it falls back to the class's own training images (flagged in metadata `source_pool_kind`, with a warning).
* **Diversity**: unique seed per image (initial noise depends only on that image's seed, so results are independent of batch composition), random source
  image, random crop/scale/aspect, flips and 90° rotations, severity sampled from the grid (`--severity_min/--severity_max/--severity_steps`, default 0.1…0.9, 9 steps),
  per-image guidance-scale jitter, optional random sub-regions (`generation.region_random_prob`).
* **Background preservation**: `composite=true` pastes the source back outside the mask (soft seam `feather`).
* `--strength < 1` starts from the noised source (SDEdit-style): more faithful, less change.
* `generation_metadata.csv` columns: `image_id, path, source_image, source_id, source_split, source_pool_kind, class, class_id, disease, species, severity,
  severity_kind, seed, mask, source_crop, checkpoint, ckpt_step, timestamp, generation_parameters (JSON), run_tag`.

## 10. Generating ≥ 10,000 images

Start small, then scale (every command is resumable with `--resume`):
```bash
python run_pipeline.py --data_dir ./data/raw --output_dir ./outputs_100  --total_images 100
python run_pipeline.py --data_dir ./data/raw --output_dir ./outputs_500  --total_images 500
python run_pipeline.py --data_dir ./data/raw --output_dir ./outputs_1k   --total_images 1000
python run_pipeline.py --data_dir ./data/raw --output_dir ./outputs      --total_images 10000
```
* `--total_images N` = number of **accepted** images wanted *after* quality control; the pipeline generates `oversample × N` (default 1.3) and runs up to
  `pipeline.max_rounds` extra rounds to fill per-class deficits, then trims to the balanced quotas keeping the best-scoring images.
* `--images_per_class 500` ⇒ total = 500 × (#classes) (20 classes → 10,000; 40 classes → 20,000), computed automatically.
* `--balance_mode uniform | inverse_frequency | proportional` (minority classes get more with `inverse_frequency`; `--balance_classes` on `generate` = uniform).
* If the target cannot be reached the run **says so** (per-class shortfall) instead of padding with rejected images.
* Already trained? `python run_pipeline.py ... --skip_train --checkpoint outputs/checkpoints/best.pt`.
* Standalone: `bash scripts/generate_10k.sh 10000`.

## 11. Quality filtering

```bash
python -m quality.quality_control --input_dir ./outputs/generated --output_dir ./outputs/filtered      # --embedder dinov2|clip|handcrafted
```
Per-image checks (thresholds in `configs/evaluation.yaml → quality`; **heuristics – calibrate on your data by looking at accepted/rejected samples**):

| Reason code | Check |
|---|---|
| `invalid_image`, `shape_mismatch`, `low_resolution` | file validity, size |
| `blurry`, `noisy_or_artifacts` | variance-of-Laplacian floor and ratio to the source's sharpness (too low / far too high) |
| `bad_exposure` | mean brightness range, saturated-pixel fraction |
| `background_modified` | mean abs. difference and changed-pixel fraction **outside** the mask (dilated by a margin) |
| `no_change_in_mask`, `over_modified` | fraction of mask pixels that changed vs. the source |
| `no_disease_signal`, `healthy_has_lesions` | colour-heuristic lesion fraction: must increase for disease classes, stay low for healthy classes |
| `unnatural_colors` | fraction of leaf pixels with blue/purple/cyan hue |
| `unrelated_to_source` | embedding cosine(source, synthetic) < `source_sim_min` (DINOv2 recommended) |
| `duplicate` | see next section |

Outputs: `quality_report.csv` (every image: all metrics, reasons), `filtered_metadata.csv`, accepted images (self-contained with `_aux/`), `quality_summary.json`
(`generated / rejected / duplicates / accepted` + reason counts), `quality_control.log`.
*Honest caveat:* the colour-based lesion check is a proxy; with `compositing` on, `background_modified` mostly validates the pipeline (run `generate` with
`--set generation.composite=false` to audit the raw model's background fidelity). `lesion_delta` vs. severity (Spearman) is a useful diagnostic of whether severity is controllable.

## 12. Deduplication

`quality/deduplicate.py`: per class, best-quality-first greedy; an image is a duplicate if it is within `dedup.hamming` bits (pHash, DCT-based, robust to noise/JPEG) of — or
has cosine ≥ `dedup.embed_sim` (DINOv2/CLIP embedding) to — an already kept image of the same class. Reported as `generated / rejected / duplicates / accepted`.
Limitation: cross-class duplicates are not looked for.

## 13. Evaluation of the generator

```bash
python -m evaluation.evaluate_generation --real_dir ./outputs/processed/test --synthetic_dir ./outputs/filtered
```
Outputs `generation_metrics_overall.json`, `generation_metrics_per_class.csv`, and plots (`fd_by_class.png`, `kid_by_class.png`,
`embedding_similarity_by_class.png`, `embedding_similarity_hist.png`, `real_vs_synthetic_distribution.png`).

| Metric | What it measures | When it is **not** meaningful |
|---|---|---|
| **FID** (`--feature_extractor inception`) | Fréchet distance between Gaussians fitted to Inception-v3 pool features of real vs synthetic sets | needs n ≳ feature dim (2048) for a stable covariance → **per-class FID with a few hundred images is biased/unreliable** (flagged `fd_reliable=False`, grey bars). Absolute values are not comparable across feature extractors, resolutions or datasets. Inception is trained on ImageNet, which is a poor proxy for leaf lesions. |
| **FD-DINOv2** (`--feature_extractor dinov2`) | the same formula on DINOv2 features | **not** comparable to published FIDs; same sample-size caveat |
| **KID** | unbiased MMD² (cubic polynomial kernel) over random subsets; mean ± std | still sensitive to content differences; use it for small samples and for relative comparisons only |
| **SSIM / PSNR vs source** | structure/pixel similarity between a synthetic image and its **source** | Computed on background (should be ≈ perfect → background preserved) and on the leaf (**expected to be lower**: disease is synthesised there). They are *not* fidelity-to-real scores; there is no ground-truth diseased version of a healthy leaf. |
| **LPIPS** (to source / within-class diversity) | learned perceptual distance | needs the `lpips` package and AlexNet weights; "to source" low ⇒ structure kept, "diversity" low ⇒ mode collapse. Not a realism score. |
| **Embedding similarity** | cosine(synthetic, its source), cosine(synthetic, class centroid / nearest real) | high similarity to the source is required (same plant) but *too* high means "nothing changed"; high similarity to real says nothing about biological plausibility |

Important evaluation caveat: synthetic images are conditioned on TRAIN-split healthy sources while the reference real set is a different set of
leaves; FD/KID therefore mix generation quality with content differences. **None of these metrics replaces the downstream test below, nor expert
inspection.**

## 14. Downstream classification

```bash
python -m evaluation.evaluate_downstream --real_dir ./outputs/processed --synthetic_dir ./outputs/filtered \
       --arch resnet50 --ratios 0 0.1 0.25 0.5 1.0 --seeds 0 1 2
python -m preprocessing.build_dataset --processed_dir ./outputs/processed --synthetic_dir ./outputs/filtered --output_dir ./outputs/combined   # folders on disk
```
* Experiments: real-only; real + 10 / 25 / 50 / 100 % synthetic (`--ratio_mode per_class`: `n_syn(c) = ratio × n_real_train(c)`; or `balance_to_max`).
* Classifier: `resnet50 | efficientnet_b0 | convnext_tiny | mobilenet_v3_large` (torchvision, ImageNet-pretrained by default; `--no_pretrained` to disable).
* Model selection uses the **REAL validation** macro-F1 (early stopping). **Test = REAL test split only**, with a hard leakage check.
* Reports accuracy, macro/weighted precision/recall/F1, per-class report (CSV), confusion matrices (PNG), mean ± std over seeds
  (`downstream_results.csv`, `downstream_summary.csv/.md`). Differences smaller than the seed std are *not* evidence of improvement.

## 15. Configuration

`configs/base.yaml` (paths, data, model), `training.yaml`, `generation.yaml`, `evaluation.yaml` (quality / evaluation / downstream), `smoke.yaml` (CPU tests).
Configs inherit with `_base_:`; override anything with `--set a.b=value` (all CLIs). The resolved config of every run is saved
(`outputs/config_used_*.yaml`) and stored inside each checkpoint. Key switches: `data.pairing`, `data.severity_mode`, `model.preset`, `model.latent`,
`model.prediction_type`, `train.precision`, `train.batch_size × train.grad_accum`, `generation.guidance_scale`, `generation.strength`, `quality.*`.

## 16. Troubleshooting

| Message / symptom | Fix |
|---|---|
| `ERROR: No class folders were found` | `--data_dir` must contain `<class>/` or `{train,val,test}/<class>/` folders with images |
| `ERROR: Only N training images were found for class X` | add images or delete the folder; minimum is `data.min_images_per_class` |
| `WARNING: No disease severity labels were found…` | expected without labels; severity will be *estimated*, or set `data.severity_mode=none` |
| `WARNING: GPU VRAM is insufficient…` / CUDA OOM | lower `data.resolution` or `train.batch_size` (raise `train.grad_accum`), `model.gradient_checkpointing=true`, `model.preset=small`, `train.precision=bf16/fp16` |
| `Loss became NaN/Inf` | lower `train.lr`, try `precision: bf16` or `fp32`, check images/masks |
| `Could not load the dinov2 model` | no internet / HF access: download once and pass a local dir via `--embedder_name`, or use `--embedder handcrafted` (weak) |
| `model.latent=true needs model.vae_path` | download a diffusers `AutoencoderKL` and set `model.vae_path` |
| Everything rejected by QC | look at `quality_report.csv`; the model is probably under-trained, or a threshold is mis-calibrated for your resolution (blur thresholds are resolution dependent) |
| Many `fallback_full` masks | segmentation failed; see `masks_preview.png`, use `grabcut`, a model, or your own masks |
| `Checkpoint does not match the model definition` | the checkpoint was trained with another `model.preset`/`latent` setting |
| YAML override like `--set quality.x=1e9` is read as a string | write `1000000000` (or `1.0e+9`) |
| Shadowed imports | run commands from the repository root; the package is `dataio/`, not `datasets/` |

## 17. GPU requirements

Designed for a single NVIDIA GPU (target: RTX 3090, 24 GB, CUDA 12.x, Python 3.10/3.11). **No memory or speed numbers have been measured**
(this repo has only been executed on CPU). The training log prints peak CUDA memory (`peak-mem x/y GB`) so you can measure it yourself.
Suggested *untested* starting points for 24 GB: `model.preset=small`, `data.resolution=256`, `train.batch_size=8`, `train.grad_accum=4`, `train.precision=fp16`
(or bf16); for 512 px use `model.latent=true` with a pretrained VAE and gradient checkpointing. Single-GPU training, mixed precision,
gradient accumulation, gradient checkpointing, resume and CUDA memory monitoring are implemented; multi-GPU is not required.

## 18. Expected folder structure

```
data/raw/<class>/*.jpg                      (your input; Format A or B)
outputs/
├── processed/                              REAL: {train,val,test}/<class>/, masks/, metadata.csv, classes.json, dataset_report.json
├── checkpoints/                            latest.pt  best.pt
├── generated/                              <class>/*.png, _aux/{sources,masks}/, generation_metadata.csv, generation.log
├── filtered/                               accepted: <class>/*.png, _aux/, filtered_metadata.csv, quality_report.csv, quality_summary.json
├── combined/                               real_only/, real_plus_syn_<ratio>/ with {train,val,test}/<class>/ + dataset_manifest.csv
├── metrics/                                generation_metrics_*.{json,csv}, training_history.csv, downstream/
├── visualizations/                         plots, grids
├── logs/                                   training.log, pipeline.log, ...
├── config_used_*.yaml   final_report.{md,json}
```
The synthetic dataset organised by class is `outputs/filtered/<class>/` (e.g. `Tomato___Late_blight/`).

## 19. Reproducibility

`common.seed.seed_everything` (python, numpy, torch, CUDA); dataset splits use `--seed`; every generated image has a **unique seed** (checked at plan time and asserted)
that fully determines its initial noise; the generation plan is deterministic given the seed; all configs and the checkpoint path are saved; each metadata row stores
`generation_parameters`, `checkpoint`, `ckpt_step`, `timestamp`. GPU kernels are not bit-wise deterministic by default (`cudnn.deterministic` is not forced), so exact pixels may differ across hardware.

## 20. Research limitations

* This is an *inspired-by* implementation; paper-level fidelity is not claimed, and no result of the paper has been reproduced.
* `pseudo_prior` pairing is an approximation of real healthy→diseased pairs (leakage of lesion colour, train/inference gap).
* Estimated severity is a colour heuristic and species/pathology dependent (yellowing, chlorosis, pests all count as "non-green"); it is not expert grading.
* Quality gates are heuristics; they reduce but do not eliminate biologically implausible images (floating lesions inside the mask, wrong lesion morphology, repeated patterns).
  Have a plant pathologist review samples per class before trusting the data.
* Species-matched healthy sources are required for plausible results; classes without healthy images fall back to editing diseased leaves.
* Synthetic data can leak the training set (memorisation); sources are real training leaves, so diversity is bounded by the number of real healthy images.
* FD/FID/KID have strong assumptions (sample size, feature extractor) – see section 13.
* Synthetic data helping on one dataset/classifier is not evidence for another; always use multiple seeds and the REAL test set.

## 21. Citation

```bibtex
@article{whensyntheticplantsgetsick2025,
  title   = {When synthetic plants get sick: Disease graded image datasets by novel regression-conditional diffusion models},
  journal = {Computers and Electronics in Agriculture},
  year    = {2025},
  volume  = {229},
  pages   = {109690},
  doi     = {10.1016/j.compag.2024.109690}
}
```
(Add the paper's author list from the publisher page.) If you use this repository, please also state that it is an independent, inspired-by implementation.

## 22. Extending to the thesis pipeline (SSL, pseudo-labels, detection)

The code is organised so that you can plug in components without touching the generator:
* **Embeddings (DINOv2/DINOv3, CLIP, SSL encoders)**: `quality/embeddings.py` – subclass `Embedder` (`embed(list of RGB uint8) -> (N,D)`) and register it in `build_embedder`; it is
  automatically usable by quality control, deduplication and evaluation.
* **Unlabeled field images**: put them in a Format-A folder, run `prepare_dataset`, embed with your encoder, cluster/pseudo-label, write the resulting labels back to
  `metadata.csv` (columns `class`, `severity`, `severity_source=pseudo`), and allow them with `data.severity_mode`. Add `pseudo` handling in `dataio.dataset.severity_allowed`.
* **Segmentation**: replace `LeafSegmenter` (section 7).
* **Semi-supervised classifier**: `evaluation/evaluate_downstream.py` builds the training list explicitly (`train_eval`) – extend with unlabeled/pseudo-labeled loaders.
* **Detection**: `generation_metadata.csv` already stores the mask of every synthetic image (`_aux/masks`); convert masks to boxes for detection datasets.
* **New datasets / pairs**: `dataio/dataset.py` (`pairing: explicit`).

## Smoke test: what it proves and what it does not

`python run_pipeline.py --data_dir tmp/raw --output_dir tmp/out --total_images 100 --preset smoke --embedder handcrafted --feature_extractor handcrafted`
runs **every stage** on procedural dummy leaves (`python scripts/make_dummy_dataset.py --out_dir tmp/raw`) with a 1.1 M-parameter model on CPU. It proves that the code runs end to end, resumes,
and that quality control rejects poor output. It says **nothing** about image quality, FID, or classification accuracy. The exact outcome of the runs performed while building this repository is
recorded in [`docs/SMOKE_TEST_LOG.md`](docs/SMOKE_TEST_LOG.md) (kept separate so that this README never contains run-specific numbers).
