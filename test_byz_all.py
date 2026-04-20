# Cartesian sweep: every (byzantine attack, aggregation) pair shares other hyperparameters.
from __future__ import print_function

import copy
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

import test_byz_p as tbp

# Must match test_byz_p.get_byz / byzantine handlers
ALL_BYZ_TYPES = ("scaling_attack", "adaptive_attack", "krum_attack", "no", "label_flipping_attack", "trim_attack")
# ALL_BYZ_TYPES = ("krum_attack","label_flipping_attack", "trim_attack")

# ALL_BYZ_TYPES = ("label_flipping_attack", "krum_attack", "trim_attack")
# Must match test_byz_p.build_arg_parser --aggregation choices
ALL_AGGREGATIONS = ("fltrust", "fedavg", "trimmed_mean", "median", "krum")
# ALL_AGGREGATIONS = ("krum",)

# Thread-local sinks: ``contextlib.redirect_stdout`` swaps *global* sys.stdout, which breaks
# when ``ThreadPoolExecutor`` runs several experiments at once (all prints share one stream).
_tls_log_out = threading.local()
_tls_log_err = threading.local()
_thread_streams_installed = False


def _tls_get_out_sink():
    return getattr(_tls_log_out, "sink", None)


def _tls_get_err_sink():
    return getattr(_tls_log_err, "sink", None)


class _ThreadLocalWriteStream(object):
    """Delegates write/flush to a per-thread sink when set, else the process default stream."""

    def __init__(self, get_sink, default_stream):
        self._get_sink = get_sink
        self._default = default_stream

    def write(self, s):
        sink = self._get_sink()
        if sink is not None:
            return sink.write(s)
        return self._default.write(s)

    def flush(self):
        sink = self._get_sink()
        if sink is not None:
            return sink.flush()
        return self._default.flush()

    def isatty(self):
        sink = self._get_sink()
        if sink is not None:
            return sink.isatty()
        return self._default.isatty()

    def fileno(self):
        sink = self._get_sink()
        if sink is not None:
            return sink.fileno()
        return self._default.fileno()

    @property
    def encoding(self):
        sink = self._get_sink()
        if sink is not None and hasattr(sink, "encoding"):
            return sink.encoding
        return getattr(self._default, "encoding", "utf-8")


def _ensure_thread_local_stdio():
    """Install once so parallel workers each log to their own file without clobbering sys.stdout."""
    global _thread_streams_installed
    if _thread_streams_installed:
        return
    sys.stdout = _ThreadLocalWriteStream(_tls_get_out_sink, sys.__stdout__)
    sys.stderr = _ThreadLocalWriteStream(_tls_get_err_sink, sys.__stderr__)
    _thread_streams_installed = True


class _redirect_stdout_stderr(object):
    """Per-thread: stdout/stderr go to *path* for the duration of the context manager."""

    def __init__(self, path):
        self.path = path
        self._logf = None
        self._prev_out = None
        self._prev_err = None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        _ensure_thread_local_stdio()
        self._logf = open(self.path, "w", buffering=1)
        self._prev_out = getattr(_tls_log_out, "sink", None)
        self._prev_err = getattr(_tls_log_err, "sink", None)
        _tls_log_out.sink = self._logf
        _tls_log_err.sink = self._logf
        return self

    def __exit__(self, *exc):
        _tls_log_out.sink = self._prev_out
        _tls_log_err.sink = self._prev_err
        if self._logf is not None:
            self._logf.flush()
            self._logf.close()
            self._logf = None
        return False


def _extract_script_flags_from_argv(argv):
    """
    Strip script-only flags from *argv* for ``test_byz_p.parse_args``.

    Returns
    -------
    new_argv : list
    log_dir : str or None
        Default ``"logs"``; set ``--log_dir`` / ``FOGGYTRUST_LOG_DIR`` overrides.
    max_workers : int or None
        ``None`` → ``ThreadPoolExecutor`` default pool size; ``1`` → sequential runs.
    """
    log_dir = os.environ.get("FOGGYTRUST_LOG_DIR") or "logs"
    max_workers = None
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--log_dir" and i + 1 < len(argv):
            log_dir = argv[i + 1]
            i += 2
        elif a.startswith("--log_dir="):
            log_dir = a.split("=", 1)[1]
            i += 1
        elif a == "--max_workers" and i + 1 < len(argv):
            max_workers = int(argv[i + 1])
            i += 2
        elif a.startswith("--max_workers="):
            max_workers = int(a.split("=", 1)[1])
            i += 1
        else:
            out.append(a)
            i += 1
    return out, log_dir, max_workers


def _byz_task(spec):
    base_args, bt, log_dir = spec
    args = copy.deepcopy(base_args)
    args.byz_type = bt
    if log_dir:
        log_path = os.path.join(log_dir, "%s.txt" % (bt,))
        with _redirect_stdout_stderr(log_path):
            return tbp.main(args)
    return tbp.main(args)


def _agg_task(spec):
    base_args, agg, log_dir = spec
    args = copy.deepcopy(base_args)
    args.aggregation = agg
    if log_dir:
        log_path = os.path.join(log_dir, "%s.txt" % (agg,))
        with _redirect_stdout_stderr(log_path):
            return tbp.main(args)
    return tbp.main(args)


def _pair_task(spec):
    base_args, bt, agg, log_dir = spec
    args = copy.deepcopy(base_args)
    args.byz_type = bt
    args.aggregation = agg
    if log_dir:
        log_path = os.path.join(log_dir, "%s__%s.txt" % (bt, agg))
        with _redirect_stdout_stderr(log_path):
            return tbp.main(args)
    return tbp.main(args)


def _asr_row(out):
    """Align attack-success series with ``test_accuracy``; NaNs when not reported."""
    y = out["test_accuracy"]
    ar = out.get("attack_success_rate")
    if ar is None:
        return np.full(y.shape[0], np.nan, dtype=np.float64)
    return np.asarray(ar, dtype=np.float64)


def _finalize_byzantine_table(byz_types, outs):
    rows = []
    asr_rows = []
    eval_x = None
    for bt, out in zip(byz_types, outs):
        x = out["eval_iteration"]
        y = out["test_accuracy"]
        if eval_x is None:
            eval_x = x
        elif not np.array_equal(eval_x, x):
            raise ValueError(
                "eval_iteration mismatch for byz_type %r (expected common grid)." % (bt,)
            )
        rows.append(y)
        asr_rows.append(_asr_row(out))
    acc = np.stack(rows, axis=0)
    asr = np.stack(asr_rows, axis=0)
    results_by_type = {bt: acc[i].copy() for i, bt in enumerate(byz_types)}
    results_asr_by_type = {bt: asr[i].copy() for i, bt in enumerate(byz_types)}
    return {
        "eval_iteration": eval_x,
        "byz_types": byz_types,
        "test_accuracy": acc,
        "attack_success_rate": asr,
        "results_by_type": results_by_type,
        "results_asr_by_type": results_asr_by_type,
    }


def _finalize_aggregation_table(aggregations, outs):
    rows = []
    asr_rows = []
    eval_x = None
    for agg, out in zip(aggregations, outs):
        x = out["eval_iteration"]
        y = out["test_accuracy"]
        if eval_x is None:
            eval_x = x
        elif not np.array_equal(eval_x, x):
            raise ValueError(
                "eval_iteration mismatch for aggregation %r (expected common grid)." % (agg,)
            )
        rows.append(y)
        asr_rows.append(_asr_row(out))
    acc = np.stack(rows, axis=0)
    asr = np.stack(asr_rows, axis=0)
    results_by_agg = {agg: acc[i].copy() for i, agg in enumerate(aggregations)}
    results_asr_by_agg = {agg: asr[i].copy() for i, agg in enumerate(aggregations)}
    return {
        "eval_iteration": eval_x,
        "aggregations": aggregations,
        "test_accuracy": acc,
        "attack_success_rate": asr,
        "results_by_agg": results_by_agg,
        "results_asr_by_agg": results_asr_by_agg,
    }


def _finalize_pairwise_table(byz_types, aggregations, outs):
    n_byz, n_agg = len(byz_types), len(aggregations)
    if len(outs) != n_byz * n_agg:
        raise ValueError(
            "expected %d outputs for pairwise sweep, got %d"
            % (n_byz * n_agg, len(outs),)
        )
    eval_x = None
    rows = []
    asr_rows = []
    for o in outs:
        x = o["eval_iteration"]
        y = o["test_accuracy"]
        if eval_x is None:
            eval_x = x
        elif not np.array_equal(eval_x, x):
            raise ValueError("eval_iteration mismatch in pairwise sweep (expected common grid).")
        rows.append(y)
        asr_rows.append(_asr_row(o))
    acc = np.reshape(np.stack(rows, axis=0), (n_byz, n_agg, -1))
    asr = np.reshape(np.stack(asr_rows, axis=0), (n_byz, n_agg, -1))
    results_by_pair = {}
    results_asr_by_pair = {}
    for i, bt in enumerate(byz_types):
        for j, agg in enumerate(aggregations):
            results_by_pair[(bt, agg)] = acc[i, j].copy()
            results_asr_by_pair[(bt, agg)] = asr[i, j].copy()
    return {
        "eval_iteration": eval_x,
        "byz_types": byz_types,
        "aggregations": aggregations,
        "test_accuracy": acc,
        "attack_success_rate": asr,
        "results_by_pair": results_by_pair,
        "results_asr_by_pair": results_asr_by_pair,
    }


def build_byzantine_timeseries_table(base_args=None, byz_types=None, log_dir=None, max_workers=None):
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
    log_dir : str or None, optional
        If set (e.g. ``"log"``), each run's stdout/stderr is written to
        ``os.path.join(log_dir, "<byz_type>.txt")`` (e.g. ``log/trim_attack.txt``).
    max_workers : int or None, optional
        ``1`` runs jobs sequentially. ``None`` uses a thread pool (default size)
        and runs one ``test_byz_p.main`` per attack in parallel.

    Returns
    -------
    dict
        ``eval_iteration`` : ndarray, shape ``(n_eval,)``, int64
        ``byz_types`` : tuple of str, length ``n_rows`` (series labels)
        ``test_accuracy`` : ndarray, shape ``(n_rows, n_eval)``, float64
        ``attack_success_rate`` : ndarray, same shape; NaN rows except for ``scaling_attack``
        ``results_by_type`` : dict str -> ndarray shape ``(n_eval,)`` (same rows, keyed)
        ``results_asr_by_type`` : dict str -> ndarray of ASR (NaN when not applicable)
    """
    if byz_types is None:
        byz_types = ALL_BYZ_TYPES
    if base_args is None:
        base_args = tbp.parse_args([])

    byz_types = tuple(byz_types)
    specs = [(base_args, bt, log_dir) for bt in byz_types]
    if max_workers == 1:
        outs = [_byz_task(s) for s in specs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            outs = list(ex.map(_byz_task, specs))

    return _finalize_byzantine_table(byz_types, outs)


def build_aggregation_timeseries_table(base_args=None, aggregations=None, log_dir=None, max_workers=None):
    """
    Run ``test_byz_p.main`` once per aggregation rule with identical hyperparameters,
    then stack test accuracies into a matrix (same layout as ``build_byzantine_timeseries_table``).

    ``--aggregation`` in ``base_args`` is overwritten for each run; ``byz_type`` and all
    other fields are preserved.

    Parameters
    ----------
    log_dir : str or None, optional
        If set, each run's stdout/stderr goes to ``os.path.join(log_dir, "<aggregation>.txt")``.
    max_workers : int or None, optional
        Same semantics as ``build_byzantine_timeseries_table``.

    Returns
    -------
    dict
        Same keys as ``build_byzantine_timeseries_table`` (``eval_iteration``,
        ``test_accuracy``, ``attack_success_rate`` with NaNs when ``byz_type`` is not
        ``scaling_attack``, ``results_by_agg``, ``results_asr_by_agg``).
    """
    if aggregations is None:
        aggregations = ALL_AGGREGATIONS
    if base_args is None:
        base_args = tbp.parse_args([])

    aggregations = tuple(aggregations)
    specs = [(base_args, agg, log_dir) for agg in aggregations]
    if max_workers == 1:
        outs = [_agg_task(s) for s in specs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            outs = list(ex.map(_agg_task, specs))

    return _finalize_aggregation_table(aggregations, outs)


def build_pairwise_timeseries_table(
    base_args=None,
    byz_types=None,
    aggregations=None,
    log_dir=None,
    max_workers=None,
):
    """
    Run ``test_byz_p.main`` for every ``(byz_type, aggregation)`` pair with shared
    hyperparameters. Log files (when ``log_dir`` is set) are
    ``<byz_type>__<aggregation>.txt`` (e.g. ``trim_attack__krum.txt``), one per pair.

    Returns
    -------
    dict
        ``eval_iteration`` : ndarray, shape ``(n_eval,)``
        ``byz_types``, ``aggregations`` : tuples of labels
        ``test_accuracy`` : ndarray, shape ``(len(byz_types), len(aggregations), n_eval)``
        ``attack_success_rate`` : ndarray, same shape (NaN except scaling_attack rows)
        ``results_by_pair`` : dict ``(byz, agg)`` -> 1d ndarray of test accuracy
        ``results_asr_by_pair`` : dict ``(byz, agg)`` -> 1d ndarray of ASR
    """
    if byz_types is None:
        byz_types = ALL_BYZ_TYPES
    if aggregations is None:
        aggregations = ALL_AGGREGATIONS
    if base_args is None:
        base_args = tbp.parse_args([])

    byz_types = tuple(byz_types)
    aggregations = tuple(aggregations)
    specs = [
        (base_args, bt, agg, log_dir)
        for bt in byz_types
        for agg in aggregations
    ]
    n_runs = len(specs)
    # Main-thread tqdm on stderr so worker log redirection (stdout+stderr → files) does not swallow it.
    _pbar_kw = {
        "total": n_runs,
        "desc": "pairwise",
        "unit": "run",
        "file": sys.stderr,
        "dynamic_ncols": True,
    }
    if max_workers == 1:
        outs = []
        with tqdm(**_pbar_kw) as pbar:
            for s in specs:
                outs.append(_pair_task(s))
                pbar.update(1)
    else:
        outs = [None] * n_runs
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(_pair_task, s): i for i, s in enumerate(specs)}
            with tqdm(**_pbar_kw) as pbar:
                for fut in as_completed(future_to_idx):
                    outs[future_to_idx[fut]] = fut.result()
                    pbar.update(1)

    return _finalize_pairwise_table(byz_types, aggregations, outs)


if __name__ == "__main__":
    argv_rest, log_dir, max_workers = _extract_script_flags_from_argv(sys.argv[1:])
    base_args = tbp.parse_args(argv_rest)
    byz_types = tuple(ALL_BYZ_TYPES)
    aggs = tuple(ALL_AGGREGATIONS)

    table = build_pairwise_timeseries_table(
        base_args, byz_types, aggs, log_dir, max_workers=max_workers
    )
    n_runs = len(byz_types) * len(aggs)
    print("Pairwise sweep — byz_types:", table["byz_types"])
    print("Pairwise sweep — aggregations:", table["aggregations"])
    print("Pairwise sweep — n_runs (|byz| x |agg|):", n_runs)
    print("Pairwise sweep — eval_iteration shape:", table["eval_iteration"].shape)
    print("Pairwise sweep — test_accuracy shape:", table["test_accuracy"].shape)
    print("Pairwise sweep — attack_success_rate shape:", table["attack_success_rate"].shape)
