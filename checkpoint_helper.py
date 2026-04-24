import os

import numpy as np


def state_checkpoint_path(checkpoint_path):
    if not checkpoint_path:
        return None
    if checkpoint_path.endswith(".params"):
        return checkpoint_path[:-7] + ".state.npz"
    return checkpoint_path + ".state.npz"


def load_model_checkpoint_if_present(net, args, ctx):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found, starting fresh:", checkpoint_path)
        return
    net.load_parameters(checkpoint_path, ctx=ctx)
    print("Loaded checkpoint:", checkpoint_path)


def save_model_checkpoint_if_requested(net, args):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        return
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    net.save_parameters(checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def load_aggregator_state_if_present(aggregator, args, ctx):
    if aggregator is None:
        return
    checkpoint_path = getattr(args, "checkpoint_path", None)
    sidecar_path = state_checkpoint_path(checkpoint_path)
    if not sidecar_path:
        return
    if not os.path.exists(sidecar_path):
        print("Aggregator state not found, starting fresh:", sidecar_path)
        return
    state = np.load(sidecar_path, allow_pickle=True)
    aggregator.load_state_dict(state, ctx)
    print("Loaded aggregator state:", sidecar_path)


def save_aggregator_state_if_requested(aggregator, args):
    if aggregator is None:
        return
    state = aggregator.state_dict()
    if state is None:
        return
    checkpoint_path = getattr(args, "checkpoint_path", None)
    sidecar_path = state_checkpoint_path(checkpoint_path)
    if not sidecar_path:
        return
    checkpoint_dir = os.path.dirname(os.path.abspath(sidecar_path))
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    np.savez(sidecar_path, **state)
    print("Saved aggregator state:", sidecar_path)
