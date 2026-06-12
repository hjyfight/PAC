"""
INRIA Person dataset loader.

Parses the original PASCAL-format .txt annotation files shipped with the
INRIA Person dataset and returns (PIL.Image, [bbox, ...]) pairs.

Each annotation line we care about looks like:
    Bounding box for object 1 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (163, 196) - (325, 707)

A single image can have multiple PASperson bboxes.
"""
import os
import re
from PIL import Image


_BBOX_RE = re.compile(
    r'Bounding box for object \d+ "PASperson".*?\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*-\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
)


def parse_inria_annotation(txt_path):
    """Return list of (xmin, ymin, xmax, ymax) for all PASperson objects."""
    bboxes = []
    with open(txt_path, 'r', encoding='latin-1') as f:
        for line in f:
            m = _BBOX_RE.search(line)
            if m:
                bboxes.append(tuple(int(v) for v in m.groups()))
    return bboxes


def load_inria(root_dir, split='train', max_images=None):
    """
    Load the INRIA Person dataset.

    Args:
        root_dir: path to INRIAPerson/ (containing Train/, Test/)
        split: 'train' or 'test'
        max_images: cap on the number of samples returned

    Returns:
        list of (PIL.Image, [(xmin, ymin, xmax, ymax), ...]) tuples
    """
    sub = 'Train' if split == 'train' else 'Test'
    # Train and Test use slightly different folder naming in the original
    # INRIA distribution: Train/pos/ vs Test/images/. Try both.
    candidate_img_dirs = [
        os.path.join(root_dir, sub, 'pos'),
        os.path.join(root_dir, sub, 'images'),
    ]
    img_dir = next((d for d in candidate_img_dirs if os.path.isdir(d)), None)
    ann_dir = os.path.join(root_dir, sub, 'annotations')

    if img_dir is None:
        raise FileNotFoundError(
            f"Image dir not found. Tried: {candidate_img_dirs}"
        )
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(f"Annotation dir not found: {ann_dir}")

    samples = []
    for fname in sorted(os.listdir(img_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in ('.png', '.jpg', '.jpeg'):
            continue
        ann_path = os.path.join(ann_dir, base + '.txt')
        if not os.path.isfile(ann_path):
            continue
        bboxes = parse_inria_annotation(ann_path)
        if not bboxes:
            continue
        img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
        samples.append((img, bboxes))
        if max_images is not None and len(samples) >= max_images:
            break

    return samples


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else \
        '/opt/data/private/hujinyu/code/CAPGen_Reproduction/INRIAPerson'
    samples = load_inria(root, split='train', max_images=5)
    print(f"Loaded {len(samples)} samples from {root}")
    for i, (img, bboxes) in enumerate(samples):
        print(f"  [{i}] size={img.size}  bboxes={bboxes}")
