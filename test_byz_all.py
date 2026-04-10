# Test all the byzantine attacks for the same setting of other hyperparameters
from __future__ import print_function

import copy
import sys

import numpy as np

import test_byz_p as tbp

# Must match test_byz_p.get_byz / byzantine handlers
ALL_BYZ_TYPES = ("no", "trim_attack", "label_flipping_attack")


def build_byzantine_timeseries_table(base_args=None, byz_types=None):
    """
    Run ``test_byz_p.main`` once per attack type with identical hyperparameters,
    then stack test accuracies into a matrix for line or image plots.

    The returned structure is matplotlib-oriented:
    - Use ``eval_iteration`` as the common x-axis (1-based training step at each eval).
    - Use ``test_accuracy`` with shape ``(len(byz_types), n_eval)``; row ``i`` is
      ``byz_types[i]`` (same as ``plt.plot(x, table['test_accuracy'][i])``).

    Parameters
    ----------
    base_args : argparse.Namespace, optional
        Hyperparameters shared across runs. Defaults to ``test_byz_p.parse_args([])``.
        When this module is run as a script, ``base_args`` is taken from
        ``test_byz_p.parse_args(sys.argv[1:])`` (same flags as ``test_byz_p.py``;
        ``--byz_type`` is overwritten per attack).
    byz_types : sequence of str, optional
        Attack identifiers accepted by ``test_byz_p.get_byz``. Defaults to ``ALL_BYZ_TYPES``.

    Returns
    -------
    dict
        ``eval_iteration`` : ndarray, shape ``(n_eval,)``, int64
        ``byz_types`` : tuple of str, length ``n_rows`` (series labels)
        ``test_accuracy`` : ndarray, shape ``(n_rows, n_eval)``, float64
        ``results_by_type`` : dict str -> ndarray shape ``(n_eval,)`` (same rows, keyed)
    """
    if byz_types is None:
        byz_types = ALL_BYZ_TYPES
    if base_args is None:
        base_args = tbp.parse_args([])

    byz_types = tuple(byz_types)
    rows = []
    eval_x = None

    for bt in byz_types:
        args = copy.deepcopy(base_args)
        args.byz_type = bt
        out = tbp.main(args)
        x = out["eval_iteration"]
        y = out["test_accuracy"]
        if eval_x is None:
            eval_x = x
        elif not np.array_equal(eval_x, x):
            raise ValueError(
                "eval_iteration mismatch for byz_type %r (expected common grid)." % (bt,)
            )
        rows.append(y)

    acc = np.stack(rows, axis=0)
    results_by_type = {bt: acc[i].copy() for i, bt in enumerate(byz_types)}
    return {
        "eval_iteration": eval_x,
        "byz_types": byz_types,
        "test_accuracy": acc,
        "results_by_type": results_by_type,
    }


if __name__ == "__main__":
    base_args = tbp.parse_args(sys.argv[1:])
    table = build_byzantine_timeseries_table(base_args=base_args)
    print("byz_types:", table["byz_types"])
    print("eval_iteration shape:", table["eval_iteration"].shape)
    print("test_accuracy shape:", table["test_accuracy"].shape)
