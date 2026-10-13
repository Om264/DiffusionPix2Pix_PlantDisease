# Smoke-test log (what was actually executed while building this repository)

**Purpose: prove that the code runs end to end. Nothing here says anything about image quality, FID, or classification accuracy.**
Data: procedural dummy leaves (`scripts/make_dummy_dataset.py`, 5 classes, 12–16 images each, green ellipses with brown spots) – not real plants.
Environment: CPU only (1 core, 3 GB RAM), Python 3.12.3, torch 2.14.1, diffusers 0.41.0, transformers 5.18.0, scipy 1.17.1, scikit-learn 1.8.0, pandas 3.0.2.

## 1. Unit tests
`pytest -q tests` → **34 passed** (final run after all fixes).

## 2. Pipeline run A – 40 training steps (64 px, `tiny` preset)
`python run_pipeline.py --data_dir tmp/raw --output_dir tmp/out --total_images 100 --preset smoke --embedder handcrafted --feature_extractor handcrafted --set train.max_steps=40 pipeline.max_rounds=1`
* All stages ran (validation/split/metadata → masks → severity → training with resume → generation → QC → balancing → dataset build → metrics → report).
* The process was killed once mid-run by the sandbox and **resumed correctly** (training continued from `latest.pt`).
* Generated 130 images; **0 accepted** (reasons: noisy_or_artifacts, over_modified, unnatural_colors, healthy_has_lesions). Visual check: the leaf interior was colour noise; the background was pixel-identical to the source (compositing works).
* The pipeline handled the empty result gracefully (per-class shortfall warning, `Not executed: …` for generation metrics, `real_plus_syn_*` identical to `real_only`).

## 3. Longer toy training – 800 steps (32 px, `tiny`, batch 16, lr 1e-3)
Training loss (min-SNR-weighted) fell from 0.28 (step 50) to 0.007 (step 800); EMA validation loss 0.0159 (step 200) → 0.0064 (step 800). Previews show leaf-shaped, leaf-coloured interiors
with some lesion-like colour but still visibly noisy texture. Generating 100 images and running QC with default thresholds (only `quality.min_size=16` for this 32-px toy resolution):
**19 accepted, 76 rejected, 5 duplicates.** This shows the gates discriminate; it does not show that the images are good.

## 4. Pipeline run B – `--skip_train`, target 40 accepted, 3 rounds
Reached 17 accepted images; the run reported the per-class shortfall (it does not pad with rejected images), `build_dataset` warned that fewer synthetic images than requested were available,
Inception-v3 features (weights fetched from GitHub) and FD/KID/SSIM/PSNR/embedding-similarity code ran (FD flagged `fd_reliable=False`: n < 2048), and a 1-epoch, non-pretrained MobileNetV3 downstream run executed for
real-only vs real+synthetic. The resulting numbers are chance-level/meaningless on this data and are intentionally not quoted.

## 5. Bugs found by the smoke tests and fixed
`class` column vs `itertuples` renaming; pandas-3 dtype of all-empty CSV columns; brittle 64-bit dHash (replaced by DCT pHash: near-copies ≤ 2 bits, distinct images ≥ 18 bits in a test);
YAML parsing `1e9` as a string (write `1000000000`); missing `--embedder` flag on `evaluate_generation`; background metrics now use a margin around the mask; per-stage `generation.log` /
`quality_control.log` inside the pipeline; warning when `build_dataset` selects no synthetic images.

## 6. NOT executed
GPU runs (memory/speed on an RTX 3090), real datasets, pretrained VAE/UNet (latent mode with real weights), DINOv2/CLIP real weights, LPIPS with real weights, DDP, any real FID/KID/accuracy/F1.
