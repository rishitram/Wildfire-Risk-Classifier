"""Shared settings and feature extraction for the wildfire risk classifier.

Used by train.py, evaluate.py and predict.py. Not meant to be run directly.

A house is labeled "prone" (1) when its fire_risk_percent is at or above
RISK_THRESHOLD, and "not prone" (0) otherwise.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
LABELED_IMAGES = ROOT / "data" / "labeled_100" / "images"
LABELED_CSV = ROOT / "data" / "labeled_100" / "labels.csv"
# split_data.py splits the 100 labeled houses into these two files.
# Both use the same image folder.
TRAIN_LABELS = ROOT / "data" / "train_labels.csv"
TEST_LABELS = ROOT / "data" / "test_labels.csv"
TRAIN_IMAGES = LABELED_IMAGES
TEST_IMAGES = LABELED_IMAGES
MODEL_PATH = ROOT / "models" / "wildfire_classifier.joblib"
RESULTS_DIR = ROOT / "results"

# ---------------------------------------------------------------- settings
RISK_THRESHOLD = 50.0  # fire_risk_percent >= this  ->  "prone"
CLASS_NAMES = ["not prone", "prone"]
RANDOM_SEED = 42

# House attributes used as features. County is left out: most counties
# appear only once or twice in 100 rows, so the model can't learn from it.
TABULAR_COLUMNS = ["state", "roof_material", "wall_material", "safety_features_vents"]


# ---------------------------------------------------------------- labels
def load_labels(csv_path: Path) -> pd.DataFrame:
    """Read a labels CSV and add a binary `prone` column."""
    df = pd.read_csv(csv_path)
    for col in TABULAR_COLUMNS:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["prone"] = (df["fire_risk_percent"] >= RISK_THRESHOLD).astype(int)
    return df


# ---------------------------------------------------------------- images
_extractor = None


def _get_extractor():
    """Pretrained ResNet18 with the classification head removed (512-d output)."""
    global _extractor
    if _extractor is None:
        import torch
        from torchvision import models, transforms

        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
        net.fc = torch.nn.Identity()
        net.eval()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _extractor = (net, transform)
    return _extractor


def image_features(image_paths) -> np.ndarray:
    """Return one 512-number feature vector per image.

    The CNN is frozen (not trained on our data): with only 100 images,
    fine-tuning it would mostly memorize the training set. Each vector is the
    average of the original image and its mirror image, which makes it a bit
    more stable.
    """
    import torch
    from PIL import Image

    net, transform = _get_extractor()
    feats = []
    with torch.no_grad():
        for path in image_paths:
            img = transform(Image.open(path).convert("RGB")).unsqueeze(0)
            both = torch.cat([img, torch.flip(img, dims=[3])])
            feats.append(net(both).mean(dim=0).numpy())
    return np.stack(feats)


def tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """The house-attribute columns the model uses."""
    return df[TABULAR_COLUMNS].copy()


def build_inputs(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Combine house attributes and image features into one table.

    Image feature columns are named img_0 ... img_511.
    """
    paths = [image_dir / name for name in df["filename"]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Images listed in the CSV but not found in {image_dir}: {missing}")
    img = pd.DataFrame(image_features(paths), columns=[f"img_{i}" for i in range(512)])
    return pd.concat([tabular_features(df).reset_index(drop=True), img], axis=1)


IMAGE_COLUMNS = [f"img_{i}" for i in range(512)]
