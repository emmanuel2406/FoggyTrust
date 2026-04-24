from __future__ import print_function

from dataclasses import dataclass

import numpy as np

# Restore deprecated np.bool for MXNet 1.x compatibility with NumPy 1.24+
if not hasattr(np, "bool"):
    np.bool = bool

from mxnet import nd


@dataclass
class FoggyTrustPartition(object):
    # Worker-local datasets and the persistent hierarchy metadata needed every round.
    worker_data: list
    worker_label: list
    worker_group: list
    group_workers: list
    worker_is_byzantine: list
    group_byzantine_counts: list
    group_trusted_sizes: list
    fog_server_data: list
    fog_server_label: list


def _get_rng(seed):
    if seed is None or seed <= 0:
        return np.random
    return np.random.RandomState(seed)


def _split_evenly(total, parts):
    if parts <= 0:
        raise ValueError("parts must be positive, got %d" % (parts,))
    base = total // parts
    extra = total % parts
    return [base + (1 if idx < extra else 0) for idx in range(parts)]


def _reshape_sample(x, ctx, dataset):
    x_ctx = x.as_in_context(ctx)
    if len(x_ctx.shape) == 2:
        return x_ctx.reshape(1, 1, int(x_ctx.shape[0]), int(x_ctx.shape[1]))
    if len(x_ctx.shape) == 3:
        return x_ctx.reshape(1, int(x_ctx.shape[0]), int(x_ctx.shape[1]), int(x_ctx.shape[2]))
    if len(x_ctx.shape) == 4:
        return x_ctx
    raise ValueError("Unsupported sample shape for %s: %r" % (dataset, tuple(x_ctx.shape)))


def _build_group_workers(num_workers, num_groups):
    if num_workers < num_groups:
        raise ValueError(
            "FoggyTrust requires at least one worker per fog group; got nworkers=%d, groups=%d"
            % (num_workers, num_groups)
        )
    # Groups are fixed once at startup so the hierarchy stays stable across rounds.
    worker_counts = _split_evenly(num_workers, num_groups)
    group_workers = []
    worker_group = [None for _ in range(num_workers)]
    worker_id = 0
    for group_id, count in enumerate(worker_counts):
        workers = []
        for _ in range(count):
            workers.append(worker_id)
            worker_group[worker_id] = group_id
            worker_id += 1
        group_workers.append(workers)
    return group_workers, worker_group


def _build_even_byzantine_layout(group_workers, nbyz):
    total_workers = sum(len(workers) for workers in group_workers)
    if nbyz < 0:
        raise ValueError("nbyz must be non-negative, got %d" % (nbyz,))
    if nbyz > total_workers:
        raise ValueError(
            "nbyz cannot exceed the number of workers; got nbyz=%d, nworkers=%d"
            % (nbyz, total_workers)
        )

    # Keep Byzantine pressure balanced across fog nodes for cleaner experiments.
    group_byzantine_counts = _split_evenly(nbyz, len(group_workers))
    worker_is_byzantine = [False for _ in range(total_workers)]

    for group_id, byz_count in enumerate(group_byzantine_counts):
        workers = group_workers[group_id]
        if byz_count > len(workers):
            raise ValueError(
                "Fog group %d has %d workers but %d Byzantine workers were assigned"
                % (group_id, len(workers), byz_count)
            )
        for worker_id in workers[:byz_count]:
            worker_is_byzantine[worker_id] = True

    return worker_is_byzantine, group_byzantine_counts


def _build_centered_label_quota(server_pc, num_labels, group_id, num_groups, p):
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1], got %r" % (p,))
    if server_pc <= 0:
        raise ValueError("server_pc must be positive, got %d" % (server_pc,))
    if num_groups <= 0:
        raise ValueError("num_groups must be positive, got %d" % (num_groups,))
    if not 0 <= group_id < num_groups:
        raise ValueError(
            "group_id must be in [0, %d), got %d" % (num_groups, group_id)
        )

    quota = [0 for _ in range(num_labels)]
    if num_labels == 1:
        quota[0] = server_pc
        return quota

    # Center each fog node on its label family; with fewer fog groups than labels, a group
    # owns labels by modulo (label_id % num_groups == group_id).
    center_labels = [label for label in range(num_labels) if (label % num_groups) == group_id]
    if not center_labels:
        center_labels = [group_id % num_labels]

    center_count = min(server_pc, max(0, int(server_pc * p)))
    center_base = center_count // len(center_labels)
    center_extra = center_count % len(center_labels)
    for idx, label in enumerate(center_labels):
        quota[label] = center_base + (1 if idx < center_extra else 0)

    remaining = server_pc - center_count
    other_labels = [label for label in range(num_labels) if label not in set(center_labels)]
    if not other_labels:
        return quota
    base = remaining // len(other_labels)
    extra = remaining % len(other_labels)
    for idx, label in enumerate(other_labels):
        quota[label] = base + (1 if idx < extra else 0)
    return quota


def _build_group_trusted_sizes(
    fog_server_pc,
    num_groups,
    fog_server_pc_mode="replicated",
):
    if fog_server_pc <= 0:
        raise ValueError("fog_server_pc must be positive, got %d" % (fog_server_pc,))
    if num_groups <= 0:
        raise ValueError("num_groups must be positive, got %d" % (num_groups,))

    mode = str(fog_server_pc_mode).strip().lower()
    if mode == "replicated":
        return [int(fog_server_pc) for _ in range(num_groups)]
    if mode == "partitioned":
        return _split_evenly(int(fog_server_pc), num_groups)
    raise ValueError(
        "fog_server_pc_mode must be one of {'replicated', 'partitioned'}, got %r"
        % (fog_server_pc_mode,)
    )


def _sample_worker_group(label_id, bias, num_labels, num_groups, rng):
    if not 0.0 <= bias <= 1.0:
        raise ValueError("bias must be in [0, 1], got %r" % (bias,))
    if num_groups <= 0:
        raise ValueError("num_groups must be positive, got %d" % (num_groups,))
    center_group = int(label_id) % int(num_groups)
    if num_groups == 1 or bias >= 1.0:
        return center_group

    # Samples mostly stay with their label-family group, with the remaining mass spread uniformly.
    probabilities = np.full((num_groups,), (1.0 - bias) / float(num_groups - 1))
    probabilities[center_group] = bias
    return int(rng.choice(num_groups, p=probabilities))


def _collect_label_buckets(train_data, ctx, dataset, num_labels):
    label_buckets = [[] for _ in range(num_labels)]
    # Bucket by true label first so we can cleanly carve out trusted fog-root datasets per family.
    for _, (data, label) in enumerate(train_data):
        for x, y in zip(data, label):
            label_id = int(np.rint(y.asnumpy()).item())
            reshaped_x = _reshape_sample(x, ctx, dataset)
            reshaped_y = y.as_in_context(ctx)
            label_buckets[label_id].append((reshaped_x, reshaped_y))
    return label_buckets


def _collect_snapshot_grouped_label_buckets(
    train_data,
    ctx,
    dataset,
    num_labels,
    snapshot_train_samples,
    project_to_group,
):
    grouped = [[[] for _ in range(num_labels)] for _ in range(len(project_to_group))]
    sample_idx = 0
    total_samples = len(snapshot_train_samples)
    for _, (data, label) in enumerate(train_data):
        for x, y in zip(data, label):
            if sample_idx >= total_samples:
                raise ValueError(
                    "Snapshot train_data produced more samples than snapshot_train_samples metadata"
                )
            rel_path, meta_label = snapshot_train_samples[sample_idx]
            project_code = str(rel_path).split("/", 1)[0].upper()
            if project_code not in project_to_group:
                raise ValueError(
                    "Snapshot sample project %r is not in configured project groups %r"
                    % (project_code, sorted(project_to_group.keys()))
                )
            label_id = int(np.rint(y.asnumpy()).item())
            if int(meta_label) != label_id:
                raise ValueError(
                    "Snapshot train_data order does not match metadata at sample %d "
                    "(metadata label=%d, loader label=%d)."
                    % (sample_idx, int(meta_label), label_id)
                )
            reshaped_x = _reshape_sample(x, ctx, dataset)
            reshaped_y = y.as_in_context(ctx)
            grouped[project_to_group[project_code]][label_id].append((reshaped_x, reshaped_y))
            sample_idx += 1
    if sample_idx != total_samples:
        if sample_idx <= 0:
            raise ValueError(
                "Snapshot train_data produced no samples; expected %d from metadata"
                % (total_samples,)
            )
        # Snapshot loaders in this repo use last_batch='rollover', which can drop the tail
        # of the manifest when len(dataset) is not divisible by batch size. In that case,
        # train_data can legitimately expose fewer samples than metadata. Keep the aligned
        # prefix and continue.
        print(
            "Snapshot train_data yielded %d/%d samples; using emitted subset "
            "(likely due to DataLoader last_batch='rollover')."
            % (sample_idx, total_samples)
        )
    return grouped


def _concat_ndarrays(values, name):
    if not values:
        raise ValueError("%s is empty" % (name,))
    return nd.concat(*values, dim=0)


def build_foggytrust_partition(
    train_data,
    bias,
    ctx,
    num_labels=10,
    num_workers=100,
    server_pc=100,
    p=0.1,
    dataset="FashionMNIST",
    seed=1,
    fog_server_pc=None,
    fog_server_pc_mode="replicated",
    nbyz=0,
    fog_num_groups=None,
    snapshot_train_samples=None,
    snapshot_projects=None,
):
    fog_server_pc = server_pc if fog_server_pc is None else fog_server_pc
    dataset_key = str(dataset).strip().lower()
    snapshot_mode = dataset_key == "snapshotsafari" and snapshot_train_samples is not None
    if snapshot_mode:
        projects = list(snapshot_projects or [])
        if not projects:
            seen = set()
            for rel_path, _ in snapshot_train_samples:
                project_code = str(rel_path).split("/", 1)[0].upper()
                if project_code in seen:
                    continue
                seen.add(project_code)
                projects.append(project_code)
        expected_groups = len(projects)
        if expected_groups <= 0:
            raise ValueError("SnapshotSafari partitioning requires at least one project group")
        if fog_num_groups is None:
            num_groups = expected_groups
        else:
            num_groups = int(fog_num_groups)
            if num_groups != expected_groups:
                raise ValueError(
                    "SnapshotSafari FoggyTrust expects one fog group per sub-dataset project; "
                    "got fog_num_groups=%d for %d projects %r"
                    % (num_groups, expected_groups, tuple(projects))
                )
        project_to_group = {project_code: idx for idx, project_code in enumerate(projects)}
    else:
        # Backward-compatible default for classic datasets (MNIST/FashionMNIST):
        # one fog node per class label (10 labels -> 10 fog nodes), with worker-group
        # assignment controlled by the original label-bias sampling path.
        if fog_num_groups is None:
            num_groups = int(num_labels)
        else:
            num_groups = int(fog_num_groups)
        project_to_group = None
    if num_groups <= 0:
        raise ValueError("fog_num_groups must be positive, got %r" % (fog_num_groups,))
    rng = _get_rng(seed)

    group_workers, worker_group = _build_group_workers(num_workers, num_groups)
    worker_is_byzantine, group_byzantine_counts = _build_even_byzantine_layout(
        group_workers, nbyz
    )

    group_trusted_sizes = _build_group_trusted_sizes(
        fog_server_pc,
        num_groups,
        fog_server_pc_mode=fog_server_pc_mode,
    )

    fog_server_data = [[] for _ in range(num_groups)]
    fog_server_label = [[] for _ in range(num_groups)]
    grouped_worker_samples = [[] for _ in range(num_groups)]

    if snapshot_mode:
        grouped_label_buckets = _collect_snapshot_grouped_label_buckets(
            train_data,
            ctx,
            dataset,
            num_labels,
            snapshot_train_samples,
            project_to_group,
        )
        for group_id in range(num_groups):
            group_samples = []
            for label_id in range(num_labels):
                group_samples.extend(grouped_label_buckets[group_id][label_id])
            rng.shuffle(group_samples)
            trusted_size = int(group_trusted_sizes[group_id])
            if trusted_size > len(group_samples):
                raise ValueError(
                    "Project-backed fog group %d has %d samples but trusted set needs %d"
                    % (
                        group_id,
                        len(group_samples),
                        trusted_size,
                    )
                )
            trusted = group_samples[:trusted_size]
            remaining = group_samples[trusted_size:]
            for x, y in trusted:
                fog_server_data[group_id].append(x)
                fog_server_label[group_id].append(y)
            grouped_worker_samples[group_id].extend(remaining)
    else:
        label_buckets = _collect_label_buckets(train_data, ctx, dataset, num_labels)
        group_label_quotas = [
            _build_centered_label_quota(
                group_trusted_sizes[group_id], num_labels, group_id, num_groups, p
            )
            for group_id in range(num_groups)
        ]

        for label_id, bucket in enumerate(label_buckets):
            rng.shuffle(bucket)
            cursor = 0

            for group_id in range(num_groups):
                take_count = group_label_quotas[group_id][label_id]
                if cursor + take_count > len(bucket):
                    raise ValueError(
                        "Not enough label-%d samples to build all fog trusted datasets "
                        "(needed %d, available %d)"
                        % (
                            label_id,
                            cursor + take_count,
                            len(bucket),
                        )
                    )
                selected = bucket[cursor : cursor + take_count]
                for x, y in selected:
                    fog_server_data[group_id].append(x)
                    fog_server_label[group_id].append(y)
                cursor += take_count

            # After reserving trusted data, the remaining client samples are assigned to one persistent fog family.
            for x, y in bucket[cursor:]:
                assigned_group = _sample_worker_group(label_id, bias, num_labels, num_groups, rng)
                grouped_worker_samples[assigned_group].append((x, y))

    worker_data = [[] for _ in range(num_workers)]
    worker_label = [[] for _ in range(num_workers)]

    for group_id, workers in enumerate(group_workers):
        samples = grouped_worker_samples[group_id]
        rng.shuffle(samples)
        if len(samples) < len(workers):
            raise ValueError(
                "Fog group %d has %d workers but only %d client samples after trusted-data allocation"
                % (group_id, len(workers), len(samples))
            )
        # Round-robin assignment keeps workers inside a fog family while distributing that family's data fairly.
        for idx, (x, y) in enumerate(samples):
            worker_id = workers[idx % len(workers)]
            worker_data[worker_id].append(x)
            worker_label[worker_id].append(y)

    for group_id in range(num_groups):
        fog_server_data[group_id] = _concat_ndarrays(
            fog_server_data[group_id], "fog_server_data[%d]" % (group_id,)
        )
        fog_server_label[group_id] = _concat_ndarrays(
            fog_server_label[group_id], "fog_server_label[%d]" % (group_id,)
        )

    for worker_id in range(num_workers):
        worker_data[worker_id] = _concat_ndarrays(
            worker_data[worker_id], "worker_data[%d]" % (worker_id,)
        )
        worker_label[worker_id] = _concat_ndarrays(
            worker_label[worker_id], "worker_label[%d]" % (worker_id,)
        )

    return FoggyTrustPartition(
        worker_data=worker_data,
        worker_label=worker_label,
        worker_group=worker_group,
        group_workers=group_workers,
        worker_is_byzantine=worker_is_byzantine,
        group_byzantine_counts=group_byzantine_counts,
        group_trusted_sizes=group_trusted_sizes,
        fog_server_data=fog_server_data,
        fog_server_label=fog_server_label,
    )
