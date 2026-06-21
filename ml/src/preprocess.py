from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TARGET_SIZE = (224, 224)
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"


@dataclass
class ImageSample:
    source: Path
    class_name: str
    bytes_hash: str
    ahash: int
    blur_score: float


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images_recursive(folder: Path) -> list[Path]:
    return [
        file
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]


def compute_bytes_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def compute_average_hash(image: Image.Image, hash_size: int = 8) -> int:
    small = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    bits = arr > arr.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def estimate_blur_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.var(gx) + np.var(gy))


def extract_leaf_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    green_mask = (hue >= 25) & (hue <= 120) & (sat >= 25) & (val >= 25)
    if not np.any(green_mask):
        return None

    ys, xs = np.where(green_mask)
    y0 = int(ys.min())
    y1 = int(ys.max())
    x0 = int(xs.min())
    x1 = int(xs.max())

    h, w = green_mask.shape
    pad_x = max(4, int((x1 - x0 + 1) * 0.1))
    pad_y = max(4, int((y1 - y0 + 1) * 0.1))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)

    if (x1 - x0) < max(20, int(w * 0.1)) or (y1 - y0) < max(20, int(h * 0.1)):
        return None

    return x0, y0, x1 + 1, y1 + 1


def preprocess_leaf_image(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    bbox = extract_leaf_bbox(image)
    if bbox is not None:
        image = image.crop(bbox)

    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = ImageOps.autocontrast(image, cutoff=1)
    return image.resize(target_size, Image.Resampling.LANCZOS)


def load_candidate(
    image_path: Path,
    class_name: str,
    min_size: int,
    blur_threshold: float,
) -> tuple[ImageSample | None, Image.Image | None, str]:
    try:
        raw = image_path.read_bytes()
        with Image.open(image_path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            width, height = image.size
            if width < min_size or height < min_size:
                return None, None, "small_or_corrupt"

            blur_score = estimate_blur_score(image)
            if blur_score < blur_threshold:
                return None, None, "blurry"

            sample = ImageSample(
                source=image_path,
                class_name=class_name,
                bytes_hash=compute_bytes_hash(raw),
                ahash=compute_average_hash(image),
                blur_score=blur_score,
            )
            return sample, image, "ok"
    except Exception:
        return None, None, "small_or_corrupt"


def integrate_raw_dirs(raw_dirs: list[Path]) -> dict[str, list[Path]]:
    class_to_paths: dict[str, list[Path]] = {}
    for raw_dir in raw_dirs:
        if not raw_dir.exists():
            print(f"Warning: raw directory not found, skipped: {raw_dir}")
            continue
        for class_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
            image_paths = list_images_recursive(class_dir)
            if not image_paths:
                print(f"Warning: class folder has no images, skipped: {class_dir}")
                continue
            class_to_paths.setdefault(class_dir.name, [])
            class_to_paths[class_dir.name].extend(image_paths)
    return class_to_paths


def select_classes(available: list[str], include_classes: list[str] | None) -> list[str]:
    if not include_classes:
        return sorted(available)
    wanted = [c.strip() for c in include_classes if c.strip()]
    selected = [c for c in sorted(available) if c in wanted]
    missing = [c for c in wanted if c not in available]
    if missing:
        print(f"Warning: requested classes not found in raw dirs: {missing}")
    return selected


def split_dataset(
    raw_dirs: list[Path],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    target_size: tuple[int, int],
    min_size: int,
    blur_threshold: float,
    duplicate_hamming_threshold: int,
    include_classes: list[str] | None,
    max_images_per_class: int | None,
) -> None:
    class_to_paths = integrate_raw_dirs(raw_dirs)
    available_classes = sorted(class_to_paths.keys())
    classes = select_classes(available_classes, include_classes)

    if not classes:
        raise ValueError(f"No class folders selected. Raw dirs checked: {raw_dirs}")

    print(f"Selected classes: {classes}")
    for split in ["train", "val", "test"]:
        clear_directory(output_dir / split)

    rng = random.Random(seed)
    global_seen_hashes: set[str] = set()
    global_ahashes: list[int] = []

    summary = {
        "kept": 0,
        "removed_small_or_corrupt": 0,
        "removed_blurry": 0,
        "removed_duplicate": 0,
        "removed_reduction_cap": 0,
    }

    for class_name in classes:
        image_paths = class_to_paths[class_name]
        if len(image_paths) < 10:
            raise ValueError(
                f"Class '{class_name}' has only {len(image_paths)} images. At least 10 are recommended."
            )

        candidates: list[tuple[ImageSample, Image.Image]] = []
        for image_path in image_paths:
            sample, image, status = load_candidate(
                image_path=image_path,
                class_name=class_name,
                min_size=min_size,
                blur_threshold=blur_threshold,
            )

            if sample is None or image is None:
                if status == "blurry":
                    summary["removed_blurry"] += 1
                else:
                    summary["removed_small_or_corrupt"] += 1
                continue

            if sample.bytes_hash in global_seen_hashes:
                summary["removed_duplicate"] += 1
                continue

            if any(
                hamming_distance(sample.ahash, seen_hash) <= duplicate_hamming_threshold
                for seen_hash in global_ahashes
            ):
                summary["removed_duplicate"] += 1
                continue

            global_seen_hashes.add(sample.bytes_hash)
            global_ahashes.append(sample.ahash)
            candidates.append((sample, image))

        if len(candidates) < 10:
            raise ValueError(
                f"Class '{class_name}' has only {len(candidates)} usable images after cleaning; "
                "collect more images or relax thresholds."
            )

        rng.shuffle(candidates)
        if max_images_per_class is not None and len(candidates) > max_images_per_class:
            summary["removed_reduction_cap"] += len(candidates) - max_images_per_class
            candidates = candidates[:max_images_per_class]

        train_count = int(len(candidates) * train_ratio)
        val_count = int(len(candidates) * val_ratio)

        split_map = {
            "train": candidates[:train_count],
            "val": candidates[train_count : train_count + val_count],
            "test": candidates[train_count + val_count :],
        }

        for split_name, items in split_map.items():
            target_dir = output_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for idx, (sample, image) in enumerate(items):
                processed = preprocess_leaf_image(image, target_size=target_size)
                out_name = f"{sample.source.stem}_{idx:04d}.jpg"
                processed.save(target_dir / out_name, format="JPEG", quality=92, optimize=True)
                summary["kept"] += 1

        print(
            f"{class_name}: train={len(split_map['train'])}, val={len(split_map['val'])}, "
            f"test={len(split_map['test'])} (usable={len(candidates)}, raw={len(image_paths)})"
        )

    print("\nCleaning summary:")
    print(f"  kept images: {summary['kept']}")
    print(f"  removed small/corrupt: {summary['removed_small_or_corrupt']}")
    print(f"  removed blurry: {summary['removed_blurry']}")
    print(f"  removed duplicate: {summary['removed_duplicate']}")
    print(f"  removed by reduction cap: {summary['removed_reduction_cap']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, integrate, preprocess, and split medicinal leaf images."
    )
    parser.add_argument(
        "--raw-dirs",
        nargs="+",
        type=Path,
        default=[DEFAULT_RAW_DIR],
        help="One or more dataset roots with class folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Path where train/val/test folders will be written",
    )
    parser.add_argument("--classes", nargs="*", default=None, help="Optional class list to include")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-width", type=int, default=DEFAULT_TARGET_SIZE[0])
    parser.add_argument("--target-height", type=int, default=DEFAULT_TARGET_SIZE[1])
    parser.add_argument("--min-size", type=int, default=128, help="Minimum width/height for usable images")
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=0.0012,
        help="Low values keep more images; high values remove more blurry images",
    )
    parser.add_argument(
        "--duplicate-hamming-threshold",
        type=int,
        default=6,
        help="Perceptual hash Hamming-distance threshold for near-duplicate removal",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Optional cap after cleaning to reduce over-represented classes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be > 0 and their sum must be < 1")

    split_dataset(
        raw_dirs=args.raw_dirs,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        target_size=(args.target_width, args.target_height),
        min_size=args.min_size,
        blur_threshold=args.blur_threshold,
        duplicate_hamming_threshold=args.duplicate_hamming_threshold,
        include_classes=args.classes,
        max_images_per_class=args.max_images_per_class,
    )
    print(f"Dataset split completed at: {args.output_dir}")


if __name__ == "__main__":
    main()
