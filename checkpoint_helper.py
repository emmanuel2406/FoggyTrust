import glob
import os
import re

import numpy as np


_CHECKPOINT_ITER_RE = re.compile(r"__iter_(\d+)$")
_CHECKPOINT_FAMILY_NITER_RE = re.compile(r"__every_10pct_of_(\d+)(?:__|$)")


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


def align_eval_outputs_for_resume(outs, asr_getter=None):
    """
    Align run outputs to a shared eval-iteration axis after checkpoint resume.

    When different runs resume from different checkpoints, each run can report
    ``eval_iteration`` on a different subset of total iterations. This helper
    merges all observed eval points and pads missing values with NaN.
    """
    if len(outs) == 0:
        empty_eval = np.asarray([], dtype=np.int64)
        empty_acc = np.empty((0, 0), dtype=np.float64)
        return empty_eval, empty_acc, empty_acc.copy()

    x_parts = []
    for out in outs:
        x = np.asarray(out["eval_iteration"], dtype=np.int64).reshape(-1)
        if x.size > 0:
            x_parts.append(x)

    if len(x_parts) == 0:
        empty_eval = np.asarray([], dtype=np.int64)
        empty_acc = np.empty((len(outs), 0), dtype=np.float64)
        return empty_eval, empty_acc, empty_acc.copy()

    eval_x = np.unique(np.concatenate(x_parts))
    iter_to_col = {int(it): idx for idx, it in enumerate(eval_x.tolist())}
    acc = np.full((len(outs), eval_x.shape[0]), np.nan, dtype=np.float64)
    asr = np.full((len(outs), eval_x.shape[0]), np.nan, dtype=np.float64)

    for row_idx, out in enumerate(outs):
        x = np.asarray(out["eval_iteration"], dtype=np.int64).reshape(-1)
        y = np.asarray(out["test_accuracy"], dtype=np.float64).reshape(-1)
        if asr_getter is None:
            row_asr = np.full(x.shape[0], np.nan, dtype=np.float64)
        else:
            row_asr = np.asarray(asr_getter(out), dtype=np.float64).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "eval_iteration/test_accuracy length mismatch for row %d: %d vs %d"
                % (row_idx, int(x.shape[0]), int(y.shape[0]))
            )
        if x.shape[0] != row_asr.shape[0]:
            raise ValueError(
                "eval_iteration/attack_success_rate length mismatch for row %d: %d vs %d"
                % (row_idx, int(x.shape[0]), int(row_asr.shape[0]))
            )
        for i, iter_v in enumerate(x):
            col = iter_to_col[int(iter_v)]
            acc[row_idx, col] = y[i]
            asr[row_idx, col] = row_asr[i]

    return eval_x, acc, asr


def _extract_checkpoint_family_niter(checkpoint_path):
    if not checkpoint_path:
        return None
    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    match = _CHECKPOINT_FAMILY_NITER_RE.search(base)
    if not match:
        return None
    return int(match.group(1))


def _rewrite_checkpoint_family_niter(checkpoint_path, niter):
    if not checkpoint_path or niter is None:
        return checkpoint_path
    return _CHECKPOINT_FAMILY_NITER_RE.sub(
        "__every_10pct_of_%d__" % (int(niter),),
        checkpoint_path,
        count=1,
    )


def find_latest_checkpoint(checkpoint_path, niter):
    base, ext = os.path.splitext(checkpoint_path)
    best_iter = -1
    best_path = None
    pattern = "%s__iter_*%s" % (base, ext)
    family_niter = _extract_checkpoint_family_niter(checkpoint_path)
    target_niter = None if niter is None else int(niter)
    if (
        family_niter is not None
        and target_niter is not None
        and int(family_niter) != int(target_niter)
    ):
        return None, -1
    for cand in glob.glob(pattern):
        stem = os.path.splitext(cand)[0]
        match = _CHECKPOINT_ITER_RE.search(stem)
        if not match:
            continue
        checkpoint_iter = int(match.group(1))
        if target_niter is not None and checkpoint_iter > int(target_niter):
            continue
        if target_niter is not None and not is_checkpoint_iteration(
            checkpoint_iter, target_niter
        ):
            continue
        if checkpoint_iter > best_iter:
            best_iter = checkpoint_iter
            best_path = cand
    return best_path, best_iter


def has_checkpoint_for_resume(checkpoint_path, niter=None):
    """True when a legacy/base or iter-suffixed checkpoint can be resumed."""
    if not checkpoint_path:
        return False
    family_niter = _extract_checkpoint_family_niter(checkpoint_path)
    if (
        niter is not None
        and family_niter is not None
        and int(family_niter) != int(niter)
    ):
        return False
    latest_path, _ = find_latest_checkpoint(checkpoint_path, niter)
    if latest_path is not None:
        return True
    return os.path.exists(checkpoint_path)


def find_completed_checkpoint(checkpoint_path, niter):
    """
    Return the iter-suffixed checkpoint path when training is already complete.

    Completion is defined as having an existing checkpoint at iteration ``niter``.
    Legacy/base checkpoints (without ``__iter_`` suffix) do not imply completion.
    """
    if not checkpoint_path or niter is None:
        return None
    latest_path, latest_iter = find_latest_checkpoint(checkpoint_path, niter)
    if latest_path is None:
        return None
    if int(latest_iter) >= int(niter):
        return latest_path
    return None


def maybe_skip_training_due_to_completed_checkpoint(args):
    """
    Detect completed checkpoints and print a dedicated skip message.

    Returns
    -------
    bool
        True when training should be skipped entirely.
    """
    checkpoint_path = getattr(args, "checkpoint_path", None)
    niter = getattr(args, "niter", None)
    family_niter = _extract_checkpoint_family_niter(checkpoint_path)
    if (
        niter is not None
        and family_niter is not None
        and int(family_niter) != int(niter)
    ):
        rewritten = _rewrite_checkpoint_family_niter(checkpoint_path, niter)
        setattr(args, "checkpoint_path", rewritten)
        checkpoint_path = rewritten
        print(
            "Checkpoint family niter mismatch while checking completion; "
            "using requested schedule path: %s"
            % (rewritten,)
        )
    completed_path = find_completed_checkpoint(checkpoint_path, niter)
    if completed_path is None:
        return False
    setattr(args, "_active_checkpoint_path", completed_path)
    print(
        "Checkpoint already complete (iter %d/%d): %s"
        % (int(niter), int(niter), completed_path)
    )
    print("Skipping run: final checkpoint exists; not loading datasets/models.")
    return True


def _checkpoint_target_path(args, iteration=None):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        return None
    if iteration is None:
        active_path = getattr(args, "_active_checkpoint_path", None)
        return active_path or checkpoint_path
    return checkpoint_path_for_iteration(checkpoint_path, iteration)


def _checkpoint_schedule_total_niter(args):
    """
    Return the total-iteration horizon used for checkpoint cadence.

    This is intentionally decoupled from any runtime "iterations left" bookkeeping.
    """
    total_niter = getattr(args, "_checkpoint_total_niter", None)
    if total_niter is not None:
        return int(total_niter)
    return int(getattr(args, "niter", 1))


def load_model_checkpoint_if_present(net, args, ctx):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        setattr(args, "_active_checkpoint_path", None)
        return 0
    target_niter = getattr(args, "niter", None)
    if target_niter is not None:
        # Pin the checkpoint cadence to total training horizon, not remaining iters.
        setattr(args, "_checkpoint_total_niter", int(target_niter))
    family_niter = _extract_checkpoint_family_niter(checkpoint_path)
    if (
        target_niter is not None
        and family_niter is not None
        and int(family_niter) != int(target_niter)
    ):
        rewritten = _rewrite_checkpoint_family_niter(checkpoint_path, target_niter)
        setattr(args, "checkpoint_path", rewritten)
        setattr(args, "_active_checkpoint_path", None)
        print(
            "Checkpoint family niter mismatch, switching to requested schedule path: "
            "%s -> %s (family=%d, requested=%d)"
            % (
                checkpoint_path,
                rewritten,
                int(family_niter),
                int(target_niter),
            )
        )
        checkpoint_path = rewritten
    latest_path, latest_iter = find_latest_checkpoint(
        checkpoint_path, target_niter
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
    schedule_niter = _checkpoint_schedule_total_niter(args)
    if iteration is not None and not is_checkpoint_iteration(iteration, schedule_niter):
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
    schedule_niter = _checkpoint_schedule_total_niter(args)
    if iteration is not None and not is_checkpoint_iteration(iteration, schedule_niter):
        return None
    target_path = _checkpoint_target_path(args, iteration=iteration)
    sidecar_path = state_checkpoint_path(target_path)
    checkpoint_dir = os.path.dirname(os.path.abspath(sidecar_path))
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    np.savez(sidecar_path, **state)
    print("Saved aggregator state:", sidecar_path)
    return sidecar_path
