# Snapshot Safari on GCP

This guide explains how to download and stage the Snapshot Safari 2024 Expansion dataset for `FoggyTrust` experiments using Google Cloud Storage.

Dataset reference:
- [Snapshot Safari 2024 Expansion (LILA)](https://lila.science/datasets/snapshot-safari-2024-expansion/)

## 1) Local directory layout

Use the following layout (paths are examples):

```text
scrap/FoggyTrust/
  data/
    snapshot/
      snapshot_safari_2024_metadata.json
      images/
        KAR/...
        KRU/...
        SER/...
```

The training scripts expect:
- `--snapshot_metadata_path` -> COCO metadata JSON
- `--snapshot_images_root` -> directory containing project-code folders

## 2) Public-access workflow (no service-account key)

The LILA bucket is publicly readable. You can pull metadata and selected projects without credentials.

```bash
# Metadata
mkdir -p data/snapshot
gcloud storage cp \
  gs://public-datasets-lila/snapshot-safari-2024-expansion/snapshot_safari_2024_metadata.zip \
  data/snapshot/
unzip -o data/snapshot/snapshot_safari_2024_metadata.zip -d data/snapshot/

# Example: sync only selected projects to keep local footprint manageable

# Domain 1 - Karoo (South Africa) - semi-desert
gcloud storage rsync --recursive \
  gs://public-datasets-lila/snapshot-safari-2024-expansion/KAR \
  data/snapshot/images/KAR

# Domain 2 - Kruger (South Africa) - savanna
gcloud storage rsync --recursive \
  gs://public-datasets-lila/snapshot-safari-2024-expansion/KRU \
  data/snapshot/images/KRU

# Domain 3 - Serengeti (Tanzania) - grassland
gcloud storage rsync --recursive \
  gs://public-datasets-lila/snapshot-safari-2024-expansion/SER \
  data/snapshot/images/SER

# If you need folders with spaces, quote both source and destination paths.
gcloud storage rsync --recursive \
  "gs://public-datasets-lila/snapshot-safari-2024-expansion/Snapshot Cameo" \
  "data/snapshot/images/Snapshot Cameo"
```

You can also use `gsutil cp`/`gsutil -m rsync` if preferred.

## 3) Service-account workflow (recommended for reproducible automation)

Use this mode for CI/HPC jobs where explicit identity is preferred.

### 3.1 Create service account and grant minimum IAM

Grant read-only storage permissions (for example, `roles/storage.objectViewer`) at the relevant bucket/project scope.

### 3.2 Store key securely

Never commit keys into the repository. Save key JSON outside versioned folders (or in a secrets manager), then export:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/secure/path/snapshot-reader-sa.json"
```

### 3.3 Authenticate and download

```bash
gcloud auth activate-service-account --key-file "$GOOGLE_APPLICATION_CREDENTIALS"
gcloud storage cp \
  gs://public-datasets-lila/snapshot-safari-2024-expansion/snapshot_safari_2024_metadata.zip \
  data/snapshot/
unzip -o data/snapshot/snapshot_safari_2024_metadata.zip -d data/snapshot/
```

Then use `gcloud storage rsync` as above for selected project folders.

## 4) Secret-handling rules

- Keep credential files outside tracked source directories whenever possible.
- Use environment variables (`GOOGLE_APPLICATION_CREDENTIALS`) instead of hard-coded paths.
- Keep only non-secret templates in git (`.env.example`).
- Rotate service-account keys regularly and revoke unused keys.

## 5) Run a Snapshot Safari experiment

```bash
python test_byz_all.py \
  --runner foggytrust \
  --dataset SnapshotSafari \
  --snapshot_metadata_path data/snapshot/snapshot_safari_2024_metadata.json \
  --snapshot_images_root data/snapshot/images \
  --snapshot_subset_projects KAR,KRU,SER \
  --snapshot_min_category_frequency 20 \
  --snapshot_max_train_samples 12000 \
  --snapshot_max_test_samples 3000 \
  --snapshot_split_seed 7
```

## 6) Optional label-map artifact

For reproducibility, write the resolved category/project index mapping:

```bash
--snapshot_label_map_out data/snapshot/snapshot_label_map.json
```

This file contains category and project index mappings used during training.
