import numpy as np
# Restore deprecated np.bool for MXNet 1.x compatibility with NumPy 1.24+
if not hasattr(np, 'bool'):
    np.bool = bool

import mxnet as mx
from mxnet import nd, autograd, gluon


def _krum_scores(samples, f):
    size = len(samples)
    neighbor_count = size - f - 2
    if neighbor_count <= 0:
        raise ValueError(
            "Krum requires at least f + 3 client updates; got n=%d, f=%d"
            % (size, f)
        )
    flattened = [sample.asnumpy().reshape(-1) for sample in samples]
    metric = []
    for idx, sample in enumerate(flattened):
        distances = [
            np.linalg.norm(sample - other)
            for j, other in enumerate(flattened)
            if j != idx
        ]
        metric.append(np.sum(np.sort(distances)[:neighbor_count]))
    return metric


def _krum(samples, f):
    metric = _krum_scores(samples, f)
    index = int(np.argmin(metric))
    return samples[index], index

def fltrust(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1
    
    # use the last gradient (server update) as the trusted source
    baseline = nd.array(param_list[-1]).squeeze()
    cos_sim = []
    new_param_list = []
    
    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = nd.array(each_param_list).squeeze()
        cos_sim.append(nd.dot(baseline, each_param_array) / (nd.norm(baseline) + 1e-9) / (nd.norm(each_param_array) + 1e-9))

        
    cos_sim = nd.stack(*cos_sim)[:-1]
    cos_sim = nd.maximum(cos_sim, 0) # relu
    normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9) # weighted trust score

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (nd.norm(param_list[i]) + 1e-9) * nd.norm(baseline))
    
    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)

    # print(global_update)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size


def fedavg(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1
    stacked = nd.concat(*[param_list[i] for i in range(n)], dim=1)
    global_update = nd.mean(stacked, axis=1, keepdims=True)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size

def trimmed_mean(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1
    # k >= f for robustness; need k < n/2 so n - 2k > 0
    k = int(f)
    max_k = max(0, (n - 1) // 2)
    if 2 * k >= n:
        k = max_k
    # (d, n): one column per honest/malicious client (server row excluded)
    stacked = nd.concat(*[param_list[i] for i in range(n)], dim=1)
    sorted_vals = nd.sort(stacked, axis=1)
    trimmed = sorted_vals[:, k : n - k]
    global_update = nd.mean(trimmed, axis=1, keepdims=True)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size


def median(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1
    stacked = nd.concat(*[param_list[i] for i in range(n)], dim=1)
    sorted_vals = nd.sort(stacked, axis=1)
    mid_lo = (n - 1) // 2
    mid_hi = n // 2
    if mid_lo == mid_hi:
        global_update = sorted_vals[:, mid_lo : mid_lo + 1]
    else:
        global_update = (sorted_vals[:, mid_lo : mid_lo + 1] + sorted_vals[:, mid_hi : mid_hi + 1]) / 2

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size


def krum(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1
    global_update, _ = _krum([param_list[i] for i in range(n)], f)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size


class ScaffoldAggregator(object):
    """
    Stateful SCAFFOLD-style aggregator for this repo's one-step local update regime.

    Notes
    -----
    - This simulator collects one local gradient per worker each round.
    - The last submitted gradient is the server/root slot (kept for API compatibility),
      but SCAFFOLD aggregation itself uses worker updates only.
    - Byzantine hooks are preserved by applying ``byz`` before control-variate correction.
    """

    def __init__(self, num_workers, total_clients=None):
        self.num_workers = int(num_workers)
        self.total_clients = int(total_clients if total_clients is not None else num_workers)
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.total_clients <= 0:
            raise ValueError("total_clients must be positive")
        self.c_global = None
        self.c_local = None

    def _ensure_state(self, template_vector, num_participants):
        if self.c_global is None:
            self.c_global = nd.zeros_like(template_vector)
        if self.c_local is None:
            self.c_local = [nd.zeros_like(template_vector) for _ in range(self.num_workers)]
        if len(self.c_local) < num_participants:
            self.c_local.extend(
                [nd.zeros_like(template_vector) for _ in range(num_participants - len(self.c_local))]
            )

    @staticmethod
    def _flatten_round_gradients(gradients):
        return [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]

    @staticmethod
    def _apply_model_update(net, update_vector, lr):
        idx = 0
        for param in net.collect_params().values():
            next_idx = idx + param.data().size
            update_slice = update_vector[idx:next_idx].reshape(param.data().shape)
            param.set_data(param.data() - lr * update_slice)
            idx = next_idx

    def step(self, gradients, net, lr, f, byz):
        """
        Perform one stateful SCAFFOLD aggregation/update step.

        Parameters mirror stateless aggregators for drop-in integration.
        """
        if len(gradients) < 2:
            raise ValueError(
                "SCAFFOLD requires at least one worker gradient and one server/root gradient"
            )

        param_list = self._flatten_round_gradients(gradients)
        param_list = byz(param_list, net, lr, f)
        num_participants = len(param_list) - 1
        if num_participants <= 0:
            raise ValueError("no worker updates available for SCAFFOLD")

        template = param_list[0]
        self._ensure_state(template, num_participants)

        corrected_updates = []
        c_deltas = []
        for worker_idx in range(num_participants):
            worker_update = param_list[worker_idx]
            corrected_updates.append(worker_update + self.c_global - self.c_local[worker_idx])

            prev_c_local = self.c_local[worker_idx]
            next_c_local = worker_update.copy()
            c_deltas.append(next_c_local - prev_c_local)
            self.c_local[worker_idx] = next_c_local

        global_update = nd.mean(nd.concat(*corrected_updates, dim=1), axis=1, keepdims=True)
        self._apply_model_update(net, global_update, lr)

        scale = float(num_participants) / float(self.total_clients)
        c_global_delta = nd.mean(nd.concat(*c_deltas, dim=1), axis=1, keepdims=True)
        self.c_global = self.c_global + scale * c_global_delta
