import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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


class VTAB1k:
    """VTAB-1k dataset loader.

    Expects data pre-organized as ImageFolder-compatible directories:
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

        # ImageNet statistics — required by pretrained ViT weights
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_path = os.path.join(data_root, task, "train")
        test_path  = os.path.join(data_root, task, "test")

        train_set = ImageFolder(root=train_path, transform=train_transform)
        test_set  = ImageFolder(root=test_path,  transform=test_transform)

        self.train = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,   # avoids tiny final batch from 1000-sample train split
        )
        self.test = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
