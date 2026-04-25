import glob
import os
import re

import numpy as np


_CHECKPOINT_ITER_RE = re.compile(r"__iter_(\d+)$")


def state_checkpoint_path(checkpoint_path):
    if not checkpoint_path:
        return None
    if checkpoint_path.endswith(".params"):
        return checkpoint_path[:-7] + ".state.npz"
    return checkpoint_path + ".state.npz"


def checkpoint_path_for_iteration(checkpoint_path, iteration):
    base, ext = os.path.splitext(checkpoint_path)
    return "%s__iter_%06d%s" % (base, int(iteration), ext)


def checkpoint_schedule(niter):
    niter_i = max(1, int(niter))
    return {
        max(1, min(niter_i, int(round(float(niter_i) * (k / 10.0)))))
        for k in range(1, 11)
    }


def is_checkpoint_iteration(iteration, niter):
    if iteration is None:
        return False
    return int(iteration) in checkpoint_schedule(niter)


def find_latest_checkpoint(checkpoint_path, niter):
    base, ext = os.path.splitext(checkpoint_path)
    best_iter = -1
    best_path = None
    pattern = "%s__iter_*%s" % (base, ext)
    for cand in glob.glob(pattern):
        stem = os.path.splitext(cand)[0]
        match = _CHECKPOINT_ITER_RE.search(stem)
        if not match:
            continue
        checkpoint_iter = int(match.group(1))
        if niter is not None and checkpoint_iter > int(niter):
            continue
        if checkpoint_iter > best_iter:
            best_iter = checkpoint_iter
            best_path = cand
    return best_path, best_iter


def has_checkpoint_for_resume(checkpoint_path, niter=None):
    """True when a legacy/base or iter-suffixed checkpoint can be resumed."""
    if not checkpoint_path:
        return False
    latest_path, _ = find_latest_checkpoint(checkpoint_path, niter)
    if latest_path is not None:
        return True
    return os.path.exists(checkpoint_path)


def _checkpoint_target_path(args, iteration=None):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        return None
    if iteration is None:
        active_path = getattr(args, "_active_checkpoint_path", None)
        return active_path or checkpoint_path
    return checkpoint_path_for_iteration(checkpoint_path, iteration)


def load_model_checkpoint_if_present(net, args, ctx):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        setattr(args, "_active_checkpoint_path", None)
        return 0
    latest_path, latest_iter = find_latest_checkpoint(
        checkpoint_path, getattr(args, "niter", None)
    )
    if latest_path is None:
        if os.path.exists(checkpoint_path):
            net.load_parameters(checkpoint_path, ctx=ctx)
            setattr(args, "_active_checkpoint_path", checkpoint_path)
            print("Loaded legacy checkpoint:", checkpoint_path)
            return 0
        setattr(args, "_active_checkpoint_path", None)
        print("Checkpoint not found, starting fresh:", checkpoint_path)
        return 0
    net.load_parameters(latest_path, ctx=ctx)
    setattr(args, "_active_checkpoint_path", latest_path)
    print("Loaded checkpoint:", latest_path)
    return int(latest_iter)


def save_model_checkpoint_if_requested(net, args, iteration=None):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        return None
    if iteration is not None and not is_checkpoint_iteration(
        iteration, getattr(args, "niter", 1)
    ):
        return None
    target_path = _checkpoint_target_path(args, iteration=iteration)
    checkpoint_dir = os.path.dirname(os.path.abspath(target_path))
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    net.save_parameters(target_path)
    setattr(args, "_active_checkpoint_path", target_path)
    print("Saved checkpoint:", target_path)
    return target_path


def load_aggregator_state_if_present(aggregator, args, ctx):
    if aggregator is None:
        return None
    checkpoint_path = _checkpoint_target_path(args)
    sidecar_path = state_checkpoint_path(checkpoint_path)
    if not sidecar_path:
        return None
    if not os.path.exists(sidecar_path):
        print("Aggregator state not found, starting fresh:", sidecar_path)
        return None
    with np.load(sidecar_path, allow_pickle=True) as state:
        aggregator.load_state_dict(state, ctx)
    print("Loaded aggregator state:", sidecar_path)
    return sidecar_path


def save_aggregator_state_if_requested(aggregator, args, iteration=None):
    if aggregator is None:
        return None
    state = aggregator.state_dict()
    if state is None:
        return None
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        return None
    if iteration is not None and not is_checkpoint_iteration(
        iteration, getattr(args, "niter", 1)
    ):
        return None
    target_path = _checkpoint_target_path(args, iteration=iteration)
    sidecar_path = state_checkpoint_path(target_path)
    checkpoint_dir = os.path.dirname(os.path.abspath(sidecar_path))
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    np.savez(sidecar_path, **state)
    print("Saved aggregator state:", sidecar_path)
    return sidecar_path
