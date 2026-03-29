import os
import random
import torchvision.transforms as transforms
import torchvision.datasets as tvd
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder


# Number of classes per VTAB-1k task
VTAB_NUM_CLASSES = {
    # --- Natural ---
    "caltech101":           102,
    "cifar100":             100,
    "dtd":                   47,
    "eurosat":               10,
    "flowers102":           102,
    "pets":                  37,
    "sun397":               397,
    "svhn":                  10,
    # --- Specialized ---
    "diabetic_retinopathy":   5,
    "kitti":                  4,
    "patch_camelyon":         2,
    "resisc45":              45,
    # --- Structured ---
    "clevr_count":            8,
    "clevr_dist":             6,
    "dmlab":                  6,
    "dsprites_loc":          16,
    "dsprites_ori":          16,
    "smallnorb_azi":         18,
    "smallnorb_ele":          9,
}

# Tasks that can be downloaded automatically via torchvision
_TORCHVISION_SUPPORTED = {
    "cifar100", "dtd", "eurosat", "flowers102", "pets", "svhn", "patch_camelyon",
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _eurosat_splits(data_root, train_tf, test_tf, seed=0):
    """EuroSAT has no predefined split — use a deterministic 80/20 split."""
    base = tvd.EuroSAT(data_root, download=True)
    indices = list(range(len(base)))
    random.Random(seed).shuffle(indices)
    k = int(0.8 * len(indices))
    tr_idx, te_idx = indices[:k], indices[k:]
    train_ds = tvd.EuroSAT(data_root, transform=train_tf, download=False)
    test_ds  = tvd.EuroSAT(data_root, transform=test_tf,  download=False)
    return Subset(train_ds, tr_idx), Subset(test_ds, te_idx)


def _load_torchvision(task, data_root, train_tf, test_tf):
    """Returns (train_dataset, test_dataset) using torchvision auto-download.

    Tasks with torchvision support:
      cifar100, dtd, eurosat, flowers102, pets, svhn, patch_camelyon

    All others return (None, None) and fall back to ImageFolder.
    """
    if task == "cifar100":
        return (
            tvd.CIFAR100(data_root, train=True,  transform=train_tf, download=True),
            tvd.CIFAR100(data_root, train=False, transform=test_tf,  download=True),
        )
    if task == "dtd":
        return (
            tvd.DTD(data_root, split="train", transform=train_tf, download=True),
            tvd.DTD(data_root, split="test",  transform=test_tf,  download=True),
        )
    if task == "eurosat":
        return _eurosat_splits(data_root, train_tf, test_tf)
    if task == "flowers102":
        return (
            tvd.Flowers102(data_root, split="train", transform=train_tf, download=True),
            tvd.Flowers102(data_root, split="test",  transform=test_tf,  download=True),
        )
    if task == "pets":
        return (
            tvd.OxfordIIITPet(data_root, split="trainval", transform=train_tf, download=True),
            tvd.OxfordIIITPet(data_root, split="test",     transform=test_tf,  download=True),
        )
    if task == "svhn":
        return (
            tvd.SVHN(data_root, split="train", transform=train_tf, download=True),
            tvd.SVHN(data_root, split="test",  transform=test_tf,  download=True),
        )
    if task == "patch_camelyon":
        return (
            tvd.PCAM(data_root, split="train", transform=train_tf, download=True),
            tvd.PCAM(data_root, split="test",  transform=test_tf,  download=True),
        )
    return None, None


class VTAB1k:
    """VTAB-1k dataset loader.

    Tasks with automatic download (via torchvision):
        cifar100, dtd, eurosat, flowers102, pets, svhn, patch_camelyon

    Remaining tasks require data pre-organized as ImageFolder-compatible directories:
        {data_root}/{task}/train/<class>/<image>
        {data_root}/{task}/test/<class>/<image>

    Uses ImageNet normalization statistics (required when using pretrained ViT weights).
    """

    def __init__(self, task: str, data_root: str, batch_size: int, num_workers: int):
        if task not in VTAB_NUM_CLASSES:
            raise ValueError(
                f"Unknown VTAB-1k task '{task}'. "
                f"Valid tasks: {sorted(VTAB_NUM_CLASSES.keys())}"
            )

        self.num_classes = VTAB_NUM_CLASSES[task]

        train_tf = _train_transform()
        test_tf  = _test_transform()

        train_set, test_set = _load_torchvision(task, data_root, train_tf, test_tf)

        if train_set is None:
            # Fall back to pre-organized ImageFolder layout
            train_set = ImageFolder(os.path.join(data_root, task, "train"), transform=train_tf)
            test_set  = ImageFolder(os.path.join(data_root, task, "test"),  transform=test_tf)

        self.train = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
