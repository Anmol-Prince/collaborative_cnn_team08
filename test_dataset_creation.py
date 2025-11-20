#!/usr/bin/env python3
"""
sample_images_infer_class_from_filename.py

Finds images under a directory (recursively), infers class = filename prefix before the first dot,
samples up to N images, copies them into output_dir/<inferred_class>/ and writes labels.csv.

Usage example (Windows):
> python sample_images_infer_class_from_filename.py --data-root "F:\\Projects\\collaborative_cnn_team08\\data\\data2\\train" --output-dir "F:\\Projects\\collaborative_cnn_team08\\data\\test_sample_1000" --n-samples 1000
"""

import argparse
import os
import random
import shutil
import csv
from pathlib import Path
from collections import defaultdict

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def find_images_and_infer_classes(root):
    """
    Walk `root` recursively, find image files and infer class from filename prefix.
    Returns dict: {class_name: [file_path, ...], ...}
    """
    root = Path(root)
    class_images = defaultdict(list)

    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            name = p.name
            # infer prefix before first dot
            if '.' in name:
                prefix = name.split('.', 1)[0]
                prefix = prefix.strip()
                if prefix == '':
                    prefix = 'unknown'
            else:
                prefix = 'unknown'
            class_images[prefix].append(str(p.resolve()))
    return class_images


def proportional_sample(class_images, total_samples, seed=None):
    """
    Sample images proportionally to class sizes but ensure at least one per class if possible.
    Returns list of tuples (class_name, original_path).
    """
    if seed is not None:
        random.seed(seed)

    classes = list(class_images.keys())
    counts = {c: len(class_images[c]) for c in classes}
    total_available = sum(counts.values())
    if total_available == 0:
        return []

    if total_samples >= total_available:
        out = []
        for c in classes:
            for p in class_images[c]:
                out.append((c, p))
        random.shuffle(out)
        return out

    # proportional allocation with minimum 1 per class where possible
    alloc = {}
    for c in classes:
        alloc[c] = int((counts[c] / total_available) * total_samples)

    for c in classes:
        if alloc[c] == 0 and counts[c] > 0:
            alloc[c] = 1

    current = sum(alloc.values())
    while current > total_samples:
        reducible = [c for c in classes if alloc[c] > 1]
        if not reducible:
            break
        c = max(reducible, key=lambda x: alloc[x])
        alloc[c] -= 1
        current -= 1

    if current < total_samples:
        fractions = []
        for c in classes:
            if counts[c] == 0:
                frac = 0
            else:
                ideal = (counts[c] / total_available) * total_samples
                frac = ideal - int(ideal)
            fractions.append((frac, c))
        fractions.sort(reverse=True)
        idx = 0
        while current < total_samples and idx < len(fractions):
            c = fractions[idx][1]
            if alloc[c] < counts[c]:
                alloc[c] += 1
                current += 1
            idx += 1
        ci = 0
        while current < total_samples:
            c = classes[ci % len(classes)]
            if alloc[c] < counts[c]:
                alloc[c] += 1
                current += 1
            ci += 1

    result = []
    for c in classes:
        k = min(alloc[c], counts[c])
        choices = random.sample(class_images[c], k=k)
        result.extend([(c, p) for p in choices])

    random.shuffle(result)
    return result


def copy_sampled_images(samples, output_dir, keep_original_names=True):
    """
    Copies sampled images to output_dir/<class_name>/ maintaining original filenames.
    Returns list of dicts: {'rel_path':..., 'label':..., 'original_path':...}
    """
    out_records = []
    output_dir = Path(output_dir)
    for class_name, src in samples:
        dest_dir = output_dir / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(src)
        if keep_original_names:
            dest_path = dest_dir / src_path.name
            # ensure unique filename if needed
            if dest_path.exists():
                base = src_path.stem
                suf = src_path.suffix
                i = 1
                while (dest_dir / f"{base}_{i}{suf}").exists():
                    i += 1
                dest_path = dest_dir / f"{base}_{i}{suf}"
        else:
            dest_path = dest_dir / f"{class_name}_{src_path.stem}{src_path.suffix}"
        shutil.copy2(src_path, dest_path)
        rel = os.path.relpath(dest_path, start=str(output_dir))
        out_records.append({'rel_path': rel.replace('\\', '/'), 'label': class_name, 'original_path': str(src_path)})
    return out_records


def write_csv(records, output_csv_path):
    keys = ['rel_path', 'label', 'original_path']
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Sample images and create CSV with inferred labels from filename prefix")
    parser.add_argument('--input', required=True, help="root folder for images")
    parser.add_argument('--output', required=True, help="directory where sampled images and CSV will be written")
    parser.add_argument('--samples', type=int, default=1000, help="number of images to sample (max)")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    args = parser.parse_args()

    data_root = args.input
    output_dir = args.output
    n_samples = args.samples
    seed = args.seed

    print("Scanning images under:", data_root)
    class_images = find_images_and_infer_classes(data_root)
    total_images = sum(len(v) for v in class_images.values())
    print("Found classes and counts (inferred from filename prefix):")
    for c, imgs in class_images.items():
        print(f"  {c}: {len(imgs)}")
    print("Total images found:", total_images)

    if total_images == 0:
        print("No images found. Exiting.")
        return

    samples = proportional_sample(class_images, n_samples, seed=seed)
    print(f"Sampling {len(samples)} images (requested {n_samples})")

    records = copy_sampled_images(samples, output_dir, keep_original_names=True)
    csv_path = os.path.join(output_dir, 'labels.csv')
    write_csv(records, csv_path)
    print("Sampled images copied to:", output_dir)
    print("CSV with labels saved to:", csv_path)


if __name__ == '__main__':
    main()
