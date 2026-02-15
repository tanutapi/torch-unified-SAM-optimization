# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A unified PyTorch library implementing Sharpness-Aware Minimization (SAM) and 7 variants with consistent training/evaluation interfaces for easy comparison and reproducibility.

## Environment Setup

```bash
conda create -n SAM python=3.9 -y
conda activate SAM
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers  # for cosine schedule with warmup
```

## Running Training

```bash
# SAM on CIFAR-10 with WideResNet-28
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type SAM --rho 0.05 --seed 1234

# Adaptive SAM (ASAM) — same class as SAM but with --adaptive flag
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type SAM --adaptive --rho 0.05 --seed 1234

# ESAM — requires --beta and --gamma
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type ESAM --rho 0.05 --beta 0.5 --gamma 0.5 --seed 1234

# FisherSAM — requires --eta
python main.py --dataset cifar10 --arch_type wideresnet28 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 --warmup_epochs 5 \
  --sam_type FisherSAM --rho 0.05 --eta 0.2 --seed 1234

# LookSAM — requires --k and --alpha
python main.py --dataset cifar100 --arch_type pyramidnet --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.05 --weight_decay 5e-4 \
  --sam_type LookSAM --rho 0.05 --k 5 --alpha 1.0 --seed 1234

# No SAM (standard SGD baseline)
python main.py --dataset cifar10 --arch_type resnet18 --optimizer sgd \
  --epochs 200 --batch_size 128 --lr 0.1 --weight_decay 5e-4 --sam_type none --seed 1234
```

**Supported values:**
- `--dataset`: `cifar10`, `cifar100`, `tinyimagenet`
- `--arch_type`: `resnet18/34/50/101/152`, `wideresnet28/34`, `pyramidnet`
- `--sam_type`: `SAM`, `ESAM`, `FisherSAM`, `FriendlySAM`, `BayesianSAM`, `GSAM`, `LookSAM`, `none`

**Output locations:**
- Logs: `src/save/{sam_type}/log/`
- Model weights: `src/save/{sam_type}/state_dict/`
- Training args: `src/save/{sam_type}/{dataset}_training_arguments.json`

## Architecture

### Training Pipeline (`main.py` → `src/utils/training_utils.py`)

```
main.py
├── initialize(seed)                          # seeds + cuDNN config
├── build_dataset(args)                       # returns DataLoaders
├── build_model(args, num_classes, device)
├── build_optimizer_and_scheduler(args, ...)
│   ├── resolve_sam_variant(name)             # case-insensitive registry lookup
│   ├── Introspect variant __init__ signature → collect matching kwargs
│   ├── Instantiate SAM variant wrapping base optimizer + model
│   └── Cosine LR schedule with warmup (via transformers)
└── per-epoch loop:
    ├── train_one_epoch(...)                  # 3 code paths (see below)
    ├── scheduler.step()
    └── evaluate(...)
```

### Three Training Code Paths in `train_one_epoch`

Different SAM variants require different call signatures:
1. **BayesianSAM**: `optimizer.step(x=inputs, y=targets, lrfactor=lrf)` — custom LR schedule
2. **ESAM**: `optimizer.step(inputs, targets, loss_fn)` — direct data integration
3. **Other SAM variants**: closure-based two-step pattern:
   ```python
   loss = criterion(model(inputs), targets)
   loss.backward()
   optimizer.first_step(zero_grad=True)
   criterion(model(inputs), targets).backward()
   optimizer.second_step(zero_grad=True)
   ```
4. **Standard SGD/Adam**: regular `loss.backward(); optimizer.step()`

### SAM Registry (`src/sam_optim/__init__.py`)

`SAMVariantRegistry` maps names → classes. ASAM reuses the `SAM` class with `adaptive=True`. The registry dynamically introspects each variant's `__init__` signature and filters kwargs accordingly — this means adding a new variant only requires adding its class and a registry entry.

### Base SAM Two-Step Optimization (`src/sam_optim/SAM.py`)

- `first_step()`: perturbs weights by `rho * grad / ||grad||`, stores originals in `state[p]["old_p"]`
- `second_step()`: restores weights, applies base optimizer step on the perturbed gradient
- BatchNorm running stats are disabled during the sharp loss evaluation (between steps)

### Adding a New SAM Variant

1. Create `src/sam_optim/NewVariant.py` implementing `first_step` / `second_step` or a custom `step`
2. Add to `_SAM_VARIANT_CLASSES` dict in `src/sam_optim/__init__.py`
3. Add variant-specific hyperparams to `main.py` argparse
4. If it needs a special training loop, add a branch in `train_one_epoch` in `src/utils/training_utils.py`
