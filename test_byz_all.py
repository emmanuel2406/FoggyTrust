# Cartesian sweep: every (byzantine attack, aggregation) pair shares other hyperparameters.
from __future__ import print_function

import copy
import importlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Must match test_byz_p.get_byz / byzantine handlers
# ALL_BYZ_TYPES = ("scaling_attack", "adaptive_attack", "krum_attack", "no", "label_flipping_attack", "trim_attack")
ALL_BYZ_TYPES = ("no", "krum_attack", "scaling_attack", "label_flipping_attack", "trim_attack")

# ALL_BYZ_TYPES = ("label_flipping_attack", "krum_attack", "trim_attack")
# Must match test_byz_p.build_arg_parser --aggregation choices
# Set to () to skip flat aggregation sweeps.
# ALL_AGGREGATIONS = ("fltrust", "fedavg", "trimmed_mean", "median", "krum", "scaffold")
ALL_AGGREGATIONS = ("scaffold",)
# Must match test_foggytrust.build_arg_parser --foggy_aggregation choices.
# Keep a single default to preserve prior one-run foggytrust sweep behavior.
FOGGYTRUST_AGGREGATIONS = ("scaffold",)

# Thread-local sinks: ``contextlib.redirect_stdout`` swaps *global* sys.stdout, which breaks
# when ``ThreadPoolExecutor`` runs several experiments at once (all prints share one stream).
_tls_log_out = threading.local()
_tls_log_err = threading.local()
_thread_streams_installed = False


def _resolve_runner_module(runner):
    """
    Resolve experiment backend:
    - flat / test_byz_p: original one-level FL setup
    - foggytrust / test_foggytrust: hierarchical FoggyTrust setup
    """
    runner_key = (runner or os.environ.get("FOGGYTRUST_RUNNER") or "flat").strip().lower()
    if runner_key in ("flat", "test_byz_p"):
        return "flat", importlib.import_module("test_byz_p")
    if runner_key in ("foggytrust", "test_foggytrust"):
        return "foggytrust", importlib.import_module("test_foggytrust")
    raise ValueError(
        "Unknown runner %r. Use one of: flat, test_byz_p, foggytrust, test_foggytrust."
        % (runner,)
    )


def _default_aggregations_for_runner(runner_name):
    if runner_name == "foggytrust":
        return FOGGYTRUST_AGGREGATIONS
    return ALL_AGGREGATIONS


def _sanitize_checkpoint_token(value):
    token = str(value).strip().replace(" ", "_")
    safe = []
    for ch in token:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "run"


def _checkpoint_path(checkpoint_dir, runner_name, sweep_name, *labels):
    if not checkpoint_dir:
        return None
    os.makedirs(checkpoint_dir, exist_ok=True)
    parts = [runner_name, sweep_name] + [_sanitize_checkpoint_token(x) for x in labels]
    filename = "__".join(parts) + ".params"
    return os.path.join(checkpoint_dir, filename)


def _is_partitioned_foggytrust(runner_name, args):
    if runner_name != "foggytrust":
        return False
    mode = str(getattr(args, "fog_server_pc_mode", "replicated")).strip().lower()
    return mode == "partitioned"


def _runner_name_tag(runner_name, args):
    if _is_partitioned_foggytrust(runner_name, args):
        return "foggytrust_p"
    return runner_name


def _aggregation_name_tag(runner_name, aggregation, args):
    if runner_name == "foggytrust":
        return "foggytrust(%s)" % (aggregation,)
    return aggregation


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
    runner : str
        ``flat`` (``test_byz_p``) or ``foggytrust`` (``test_foggytrust``).
    checkpoint_dir : str or None
        Default ``"checkpoints"``. Each sweep run loads/saves one matching checkpoint file.
    """
    log_dir = os.environ.get("FOGGYTRUST_LOG_DIR") or "logs"
    checkpoint_dir = os.environ.get("FOGGYTRUST_CHECKPOINT_DIR") or "checkpoints"
    max_workers = None
    runner = os.environ.get("FOGGYTRUST_RUNNER") or "flat"
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
        elif a == "--runner" and i + 1 < len(argv):
            runner = argv[i + 1]
            i += 2
        elif a.startswith("--runner="):
            runner = a.split("=", 1)[1]
            i += 1
        elif a == "--checkpoint_dir" and i + 1 < len(argv):
            checkpoint_dir = argv[i + 1]
            i += 2
        elif a.startswith("--checkpoint_dir="):
            checkpoint_dir = a.split("=", 1)[1]
            i += 1
        else:
            out.append(a)
            i += 1
    if checkpoint_dir and checkpoint_dir.lower() in ("none", "null", "off", "false", "0"):
        checkpoint_dir = None
    return out, log_dir, checkpoint_dir, max_workers, runner


def _byz_task(spec):
    runner_name, runner_module, base_args, bt, log_dir, checkpoint_dir = spec
    args = copy.deepcopy(base_args)
    args.byz_type = bt
    runner_tag = _runner_name_tag(runner_name, args)
    args.checkpoint_path = _checkpoint_path(checkpoint_dir, runner_tag, "byz", bt)
    if log_dir:
        log_path = os.path.join(log_dir, "%s.txt" % (bt,))
        with _redirect_stdout_stderr(log_path):
            return runner_module.main(args)
    return runner_module.main(args)


def _agg_task(spec):
    runner_name, runner_module, base_args, agg, log_dir, checkpoint_dir = spec
    args = copy.deepcopy(base_args)
    if runner_name == "foggytrust":
        args.aggregation = "foggytrust"
        args.foggy_aggregation = agg
    else:
        args.aggregation = agg
    runner_tag = _runner_name_tag(runner_name, args)
    agg_tag = _aggregation_name_tag(runner_name, agg, args)
    args.checkpoint_path = _checkpoint_path(checkpoint_dir, runner_tag, "agg", agg_tag)
    if log_dir:
        log_path = os.path.join(log_dir, "%s.txt" % (agg_tag,))
        with _redirect_stdout_stderr(log_path):
            return runner_module.main(args)
    return runner_module.main(args)


def _pair_task(spec):
    runner_name, runner_module, base_args, bt, agg, log_dir, checkpoint_dir = spec
    args = copy.deepcopy(base_args)
    args.byz_type = bt
    if runner_name == "foggytrust":
        args.aggregation = "foggytrust"
        args.foggy_aggregation = agg
    else:
        args.aggregation = agg
    runner_tag = _runner_name_tag(runner_name, args)
    agg_tag = _aggregation_name_tag(runner_name, agg, args)
    args.checkpoint_path = _checkpoint_path(checkpoint_dir, runner_tag, "pair", bt, agg_tag)
    if log_dir:
        log_path = os.path.join(log_dir, "%s__%s.txt" % (bt, agg_tag))
        with _redirect_stdout_stderr(log_path):
            return runner_module.main(args)
    return runner_module.main(args)


def _asr_row(out):
    """Align attack-success series with ``test_accuracy``; NaNs when not reported."""
    y = out["test_accuracy"]
    ar = out.get("attack_success_rate")
    if ar is None:
        return np.full(y.shape[0], np.nan, dtype=np.float64)
    return np.asarray(ar, dtype=np.float64)


def _finalize_byzantine_table(byz_types, outs):
    if len(byz_types) == 0:
        empty_eval = np.asarray([], dtype=np.int64)
        empty_acc = np.empty((0, 0), dtype=np.float64)
        return {
            "eval_iteration": empty_eval,
            "byz_types": tuple(),
            "test_accuracy": empty_acc,
            "attack_success_rate": empty_acc.copy(),
            "results_by_type": {},
            "results_asr_by_type": {},
        }

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
    if len(aggregations) == 0:
        empty_eval = np.asarray([], dtype=np.int64)
        empty_acc = np.empty((0, 0), dtype=np.float64)
        return {
            "eval_iteration": empty_eval,
            "aggregations": tuple(),
            "test_accuracy": empty_acc,
            "attack_success_rate": empty_acc.copy(),
            "results_by_agg": {},
            "results_asr_by_agg": {},
        }

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
    if n_byz == 0 or n_agg == 0:
        empty_eval = np.asarray([], dtype=np.int64)
        empty_acc = np.empty((n_byz, n_agg, 0), dtype=np.float64)
        return {
            "eval_iteration": empty_eval,
            "byz_types": byz_types,
            "aggregations": aggregations,
            "test_accuracy": empty_acc,
            "attack_success_rate": empty_acc.copy(),
            "results_by_pair": {},
            "results_asr_by_pair": {},
        }

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


def build_byzantine_timeseries_table(
    base_args=None,
    byz_types=None,
    log_dir=None,
    checkpoint_dir="checkpoints",
    max_workers=None,
    runner="flat",
):
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
    checkpoint_dir : str or None, optional
        If set, each run loads/saves params at
        ``os.path.join(checkpoint_dir, "<runner>__byz__<byz_type>.params")``.
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
    runner_name, runner_module = _resolve_runner_module(runner)
    if base_args is None:
        base_args = runner_module.parse_args([])

    byz_types = tuple(byz_types)
    specs = [
        (runner_name, runner_module, base_args, bt, log_dir, checkpoint_dir)
        for bt in byz_types
    ]
    if max_workers == 1:
        outs = [_byz_task(s) for s in specs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            outs = list(ex.map(_byz_task, specs))

    return _finalize_byzantine_table(byz_types, outs)


def build_aggregation_timeseries_table(
    base_args=None,
    aggregations=None,
    log_dir=None,
    checkpoint_dir="checkpoints",
    max_workers=None,
    runner="flat",
):
    """
    Run ``test_byz_p.main`` once per aggregation rule with identical hyperparameters,
    then stack test accuracies into a matrix (same layout as ``build_byzantine_timeseries_table``).

    ``--aggregation`` in ``base_args`` is overwritten for each run; ``byz_type`` and all
    other fields are preserved.

    Parameters
    ----------
    log_dir : str or None, optional
        If set, each run's stdout/stderr goes to ``os.path.join(log_dir, "<aggregation>.txt")``.
    checkpoint_dir : str or None, optional
        If set, each run loads/saves params at
        ``os.path.join(checkpoint_dir, "<runner>__agg__<aggregation>.params")``.
    max_workers : int or None, optional
        Same semantics as ``build_byzantine_timeseries_table``.
    runner : :str, optional
        ``flat`` (``test_byz_p``) or ``foggytrust`` (``test_foggytrust``).
    Returns
    -------
    dict
        Same keys as ``build_byzantine_timeseries_table`` (``eval_iteration``,
        ``test_accuracy``, ``attack_success_rate`` with NaNs when ``byz_type`` is not
        ``scaling_attack``, ``results_by_agg``, ``results_asr_by_agg``).
    """
    runner_name, runner_module = _resolve_runner_module(runner)
    if aggregations is None:
        aggregations = _default_aggregations_for_runner(runner_name)
    if base_args is None:
        base_args = runner_module.parse_args([])

    aggregations = tuple(aggregations)
    specs = [
        (runner_name, runner_module, base_args, agg, log_dir, checkpoint_dir)
        for agg in aggregations
    ]
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
    checkpoint_dir="checkpoints",
    max_workers=None,
    runner="flat",
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
    runner_name, runner_module = _resolve_runner_module(runner)
    if aggregations is None:
        aggregations = _default_aggregations_for_runner(runner_name)
    if base_args is None:
        base_args = runner_module.parse_args([])

    byz_types = tuple(byz_types)
    aggregations = tuple(aggregations)
    specs = [
        (runner_name, runner_module, base_args, bt, agg, log_dir, checkpoint_dir)
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


def build_scaffold_pairwise_timeseries_table(
    base_args=None,
    byz_types=None,
    aggregations=None,
    log_dir=None,
    checkpoint_dir="checkpoints",
    max_workers=None,
    runner="flat",
):
    """
    Encapsulated pairwise sweep for stateful SCAFFOLD aggregation.

    Defaults to the flat runner and ``("scaffold",)`` aggregation while reusing the
    existing pairwise output contract.
    """
    if aggregations is None:
        aggregations = ("scaffold",)
    return build_pairwise_timeseries_table(
        base_args=base_args,
        byz_types=byz_types,
        aggregations=aggregations,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_workers=max_workers,
        runner=runner,
    )


if __name__ == "__main__":
    argv_rest, log_dir, checkpoint_dir, max_workers, runner = _extract_script_flags_from_argv(
        sys.argv[1:]
    )
    runner_name, runner_module = _resolve_runner_module(runner)
    base_args = runner_module.parse_args(argv_rest)
    byz_types = tuple(ALL_BYZ_TYPES)
    aggs = tuple(_default_aggregations_for_runner(runner_name))

    table = build_pairwise_timeseries_table(
        base_args,
        byz_types,
        aggs,
        log_dir,
        checkpoint_dir=checkpoint_dir,
        max_workers=max_workers,
        runner=runner_name,
    )
    n_runs = len(byz_types) * len(aggs)
    print("Pairwise sweep — runner:", runner_name)
    print("Pairwise sweep — byz_types:", table["byz_types"])
    print("Pairwise sweep — aggregations:", table["aggregations"])
    print("Pairwise sweep — n_runs (|byz| x |agg|):", n_runs)
    print("Pairwise sweep — checkpoint_dir:", checkpoint_dir)
    print("Pairwise sweep — eval_iteration shape:", table["eval_iteration"].shape)
    print("Pairwise sweep — test_accuracy shape:", table["test_accuracy"].shape)
    print("Pairwise sweep — attack_success_rate shape:", table["attack_success_rate"].shape)
