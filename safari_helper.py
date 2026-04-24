from __future__ import print_function

import hashlib
import json
import os

import mxnet as mx
import numpy as np
from mxnet import nd


def add_snapshot_safari_args(parser):
    parser.add_argument(
        "--snapshot_metadata_path",
        type=str,
        default="data/snapshot/snapshot_safari_2024_metadata.json",
        help="path to Snapshot Safari COCO metadata JSON",
    )
    parser.add_argument(
        "--snapshot_images_root",
        type=str,
        default="data/snapshot/images",
        help="root directory containing Snapshot Safari image subfolders",
    )
    parser.add_argument(
        "--snapshot_subset_projects",
        type=str,
        default="KAR,KRU,SER",
        help="comma-separated Snapshot project prefixes to include (e.g. KAR,KRU,SER)",
    )
    parser.add_argument(
        "--snapshot_min_category_frequency",
        type=int,
        default=20,
        help="minimum image frequency per category after filtering to selected projects",
    )
    parser.add_argument(
        "--snapshot_max_train_samples",
        type=int,
        default=12000,
        help="max number of Snapshot training samples (<=0 means unlimited)",
    )
    parser.add_argument(
        "--snapshot_max_test_samples",
        type=int,
        default=3000,
        help="max number of Snapshot test samples (<=0 means unlimited)",
    )
    parser.add_argument(
        "--snapshot_split_seed",
        type=int,
        default=7,
        help="random seed for Snapshot train/test split",
    )
    parser.add_argument(
        "--snapshot_label_map_out",
        type=str,
        default="",
        help="optional output JSON path for Snapshot resolved label mapping",
    )
    return parser


def _snapshot_project_set(projects_csv):
    projects = [part.strip().upper() for part in str(projects_csv).split(",")]
    return set([part for part in projects if part])


def _strip_trailing_comma(line):
    return line.strip().rstrip(",")


def _extract_json_string_value(line):
    _, rhs = line.split(":", 1)
    return json.loads(_strip_trailing_comma(rhs))


def _extract_json_int_value(line):
    _, rhs = line.split(":", 1)
    return int(json.loads(_strip_trailing_comma(rhs)))


def _snapshot_cache_file(args):
    metadata_path = os.path.abspath(args.snapshot_metadata_path)
    images_root = os.path.abspath(args.snapshot_images_root)
    cache_root = os.path.join(os.path.dirname(metadata_path), ".snapshot_cache")
    os.makedirs(cache_root, exist_ok=True)
    cache_key = {
        "metadata_path": metadata_path,
        "metadata_mtime_ns": os.path.getmtime(metadata_path),
        "images_root": images_root,
        "subset_projects": sorted(list(_snapshot_project_set(args.snapshot_subset_projects))),
        "min_category_frequency": int(args.snapshot_min_category_frequency),
        "max_train_samples": int(args.snapshot_max_train_samples),
        "max_test_samples": int(args.snapshot_max_test_samples),
        "snapshot_split_seed": int(args.snapshot_split_seed),
    }
    token = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_root, "snapshot_%s.json" % (token,))


def _iter_snapshot_section_lines(metadata_path, section_name):
    section_header = '"%s": [' % (section_name,)
    in_section = False
    with open(metadata_path, "r", encoding="utf-8") as meta_file:
        for raw_line in meta_file:
            line = raw_line.strip()
            if not in_section:
                if line.startswith(section_header):
                    in_section = True
                continue
            if line.startswith("]"):
                return
            yield line


def _build_snapshot_manifest(args):
    metadata_path = os.path.abspath(args.snapshot_metadata_path)
    images_root = os.path.abspath(args.snapshot_images_root)
    subset_projects = _snapshot_project_set(args.snapshot_subset_projects)
    if not subset_projects:
        raise ValueError("snapshot_subset_projects must include at least one project code")

    category_name_by_id = {}
    current_category_name = None
    for line in _iter_snapshot_section_lines(metadata_path, "categories"):
        if line.startswith('"name"'):
            current_category_name = _extract_json_string_value(line)
        elif line.startswith('"id"'):
            category_id = _extract_json_int_value(line)
            if current_category_name is None:
                raise ValueError("Malformed categories section in Snapshot metadata")
            category_name_by_id[category_id] = current_category_name
            current_category_name = None

    image_to_category = {}
    category_counts = {}
    current_image_id = None
    for line in _iter_snapshot_section_lines(metadata_path, "annotations"):
        if line.startswith('"image_id"'):
            current_image_id = _extract_json_string_value(line)
        elif line.startswith('"category_id"'):
            category_id = _extract_json_int_value(line)
            if current_image_id is None:
                continue
            project_code = current_image_id.split("/", 1)[0].upper()
            if project_code not in subset_projects:
                continue
            if current_image_id in image_to_category:
                continue
            image_to_category[current_image_id] = category_id
            category_counts[category_id] = category_counts.get(category_id, 0) + 1
            current_image_id = None

    min_frequency = max(1, int(args.snapshot_min_category_frequency))
    kept_category_ids = sorted(
        category_id
        for category_id, count in category_counts.items()
        if count >= min_frequency
    )
    if not kept_category_ids:
        raise ValueError(
            "No Snapshot categories meet min frequency %d for selected projects %s"
            % (min_frequency, ",".join(sorted(list(subset_projects))))
        )

    label_to_index = {
        int(category_id): idx for idx, category_id in enumerate(kept_category_ids)
    }
    class_names = [
        category_name_by_id.get(category_id, "category_%d" % (category_id,))
        for category_id in kept_category_ids
    ]
    candidates = []
    for image_id, category_id in image_to_category.items():
        if category_id not in label_to_index:
            continue
        candidates.append((image_id, int(label_to_index[category_id])))

    rng = np.random.RandomState(seed=int(args.snapshot_split_seed))
    rng.shuffle(candidates)

    max_train = int(args.snapshot_max_train_samples)
    max_test = int(args.snapshot_max_test_samples)
    max_train = None if max_train <= 0 else max_train
    max_test = None if max_test <= 0 else max_test
    max_total = None
    if max_train is not None and max_test is not None:
        max_total = (max_train + max_test) * 4

    existing = []
    for rel_path, label_idx in candidates:
        abs_path = os.path.join(images_root, rel_path)
        if not os.path.exists(abs_path):
            continue
        existing.append((rel_path, int(label_idx)))
        if max_total is not None and len(existing) >= max_total:
            break

    if len(existing) < 2:
        raise ValueError("Not enough Snapshot samples with local images under %s" % (images_root,))

    split_idx = max(1, int(len(existing) * 0.8))
    split_idx = min(split_idx, len(existing) - 1)
    train_samples = existing[:split_idx]
    test_samples = existing[split_idx:]
    if max_train is not None:
        train_samples = train_samples[:max_train]
    if max_test is not None:
        test_samples = test_samples[:max_test]
    if not train_samples or not test_samples:
        raise ValueError("Snapshot split produced an empty train/test set; adjust limits")

    return {
        "num_labels": len(kept_category_ids),
        "class_names": class_names,
        "category_ids": kept_category_ids,
        "train_samples": train_samples,
        "test_samples": test_samples,
    }


def load_snapshot_manifest(args):
    cache_path = _snapshot_cache_file(args)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            return json.load(cache_file)
    manifest = _build_snapshot_manifest(args)
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(manifest, cache_file)
    return manifest


class SnapshotSafariDataset(mx.gluon.data.Dataset):
    def __init__(self, samples, images_root, target_hw):
        self.samples = [(str(path), int(label)) for path, label in samples]
        self.images_root = images_root
        self.target_h = int(target_hw[0])
        self.target_w = int(target_hw[1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        abs_path = os.path.join(self.images_root, rel_path)
        image = mx.image.imread(abs_path, flag=1)
        image = mx.image.imresize(image, self.target_w, self.target_h)
        image = nd.transpose(image.astype(np.float32), (2, 0, 1)) / 255.0
        return image, nd.array([label], dtype=np.float32)[0]


def auto_snapshot_resize_hw(samples, images_root):
    if not samples:
        raise ValueError("Cannot infer Snapshot image shape from an empty sample list")
    probe_rel = samples[0][0]
    probe_abs = os.path.join(images_root, probe_rel)
    probe_img = mx.image.imread(probe_abs, flag=1)
    src_h = int(probe_img.shape[0])
    src_w = int(probe_img.shape[1])
    short_side = max(1, min(src_h, src_w))

    # Auto-pick a stable training size from source resolution tiers.
    if short_side >= 1024:
        target_short = 96
    elif short_side >= 512:
        target_short = 128
    elif short_side >= 256:
        target_short = 160
    else:
        target_short = short_side

    scale = float(target_short) / float(short_side)
    target_h = max(32, int(round(src_h * scale)))
    target_w = max(32, int(round(src_w * scale)))
    return target_h, target_w


def write_snapshot_label_map_if_requested(args, manifest):
    if not args.snapshot_label_map_out:
        return
    label_map_dir = os.path.dirname(os.path.abspath(args.snapshot_label_map_out))
    if label_map_dir:
        os.makedirs(label_map_dir, exist_ok=True)
    with open(args.snapshot_label_map_out, "w", encoding="utf-8") as out_file:
        json.dump(
            {
                "category_ids": manifest["category_ids"],
                "class_names": manifest["class_names"],
                "num_labels": manifest["num_labels"],
            },
            out_file,
            indent=2,
        )


def load_snapshot_safari_data(args, batch_size=256, last_batch="rollover"):
    manifest = load_snapshot_manifest(args)
    images_root = os.path.abspath(args.snapshot_images_root)
    target_hw = auto_snapshot_resize_hw(manifest["train_samples"], images_root)
    print(
        "SnapshotSafari resize auto-detected: %dx%d"
        % (int(target_hw[0]), int(target_hw[1]))
    )
    write_snapshot_label_map_if_requested(args, manifest)
    train_dataset = SnapshotSafariDataset(
        manifest["train_samples"],
        images_root,
        target_hw,
    )
    test_dataset = SnapshotSafariDataset(
        manifest["test_samples"],
        images_root,
        target_hw,
    )
    train_data = mx.gluon.data.DataLoader(
        train_dataset, batch_size=int(batch_size), shuffle=True, last_batch=last_batch
    )
    test_data = mx.gluon.data.DataLoader(
        test_dataset, batch_size=int(batch_size), shuffle=False, last_batch=last_batch
    )
    return train_data, test_data, {"num_labels": int(manifest["num_labels"])}
