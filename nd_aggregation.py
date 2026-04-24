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


def _flatten_round_gradients(gradients):
    return [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]


def _apply_model_update(net, update_vector, lr):
    idx = 0
    for param in net.collect_params().values():
        next_idx = idx + param.data().size
        update_slice = update_vector[idx:next_idx].reshape(param.data().shape)
        param.set_data(param.data() - lr * update_slice)
        idx = next_idx


def _nd_to_numpy(value):
    return None if value is None else value.asnumpy()


def _numpy_to_nd(value, ctx):
    if value is None:
        return None
    return nd.array(value, ctx=ctx)


class FedAdamCore(object):
    """
    Stateful FedAdam server optimizer over flattened update vectors.

    The input is the already-aggregated update for the current round. The returned
    vector is the adaptive server step, including the FedAdam server learning rate.
    """

    def __init__(self, eta=0.1, beta_1=0.9, beta_2=0.99, tau=1e-3):
        self.eta = float(eta)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.tau = float(tau)
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if not 0.0 <= self.beta_1 < 1.0:
            raise ValueError("beta_1 must be in [0, 1)")
        if not 0.0 <= self.beta_2 < 1.0:
            raise ValueError("beta_2 must be in [0, 1)")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        self.m_t = None
        self.v_t = None
        self.round_idx = 0

    def _ensure_state(self, template_vector):
        if self.m_t is None:
            self.m_t = nd.zeros_like(template_vector)
        if self.v_t is None:
            self.v_t = nd.zeros_like(template_vector)

    def step(self, aggregated_update):
        self._ensure_state(aggregated_update)
        self.round_idx += 1

        self.m_t = self.beta_1 * self.m_t + (1.0 - self.beta_1) * aggregated_update
        self.v_t = self.beta_2 * self.v_t + (1.0 - self.beta_2) * nd.square(
            aggregated_update
        )

        eta_norm = (
            self.eta
            * np.sqrt(1.0 - np.power(self.beta_2, float(self.round_idx)))
            / (1.0 - np.power(self.beta_1, float(self.round_idx)))
        )
        return eta_norm * self.m_t / (nd.sqrt(self.v_t) + self.tau)

    def state_dict(self):
        if self.m_t is None or self.v_t is None:
            return None
        return {
            "aggregator_kind": np.asarray(["fedadam_core"], dtype=object),
            "tensor_shape": np.asarray(self.m_t.shape, dtype=np.int64),
            "m_t": _nd_to_numpy(self.m_t),
            "v_t": _nd_to_numpy(self.v_t),
            "round_idx": np.asarray([int(self.round_idx)], dtype=np.int64),
        }

    def load_state_dict(self, state, ctx):
        if str(state["aggregator_kind"][0]) != "fedadam_core":
            raise ValueError("FedAdamCore state kind mismatch")
        saved_shape = tuple(int(x) for x in state["tensor_shape"].tolist())
        m_t = _numpy_to_nd(state["m_t"], ctx)
        v_t = _numpy_to_nd(state["v_t"], ctx)
        if tuple(m_t.shape) != saved_shape or tuple(v_t.shape) != saved_shape:
            raise ValueError("FedAdamCore tensor shape mismatch")
        self.m_t = m_t
        self.v_t = v_t
        self.round_idx = int(state["round_idx"][0])

def fltrust(gradients, net, lr, f, byz):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    
    param_list = _flatten_round_gradients(gradients)
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
    param_list = _flatten_round_gradients(gradients)
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
    param_list = _flatten_round_gradients(gradients)
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
    param_list = _flatten_round_gradients(gradients)
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
    param_list = _flatten_round_gradients(gradients)
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1
    global_update, _ = _krum([param_list[i] for i in range(n)], f)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size


class FedAdamAggregator(object):
    """
    Stateful FedAdam aggregator for this repo's one-step local update regime.

    Notes
    -----
    - Worker gradients are averaged uniformly, matching the current simulator's
      existing aggregation convention.
    - The final server/root slot is ignored for the FedAdam server step, just
      as other non-FLTrust aggregators ignore it.
    - The FedAdam core returns an adaptive server update vector, which is then
      subtracted directly from the model parameters.
    """

    def __init__(self, num_workers, eta=0.1, beta_1=0.9, beta_2=0.99, tau=1e-3):
        self.num_workers = int(num_workers)
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self.core = FedAdamCore(
            eta=eta, beta_1=beta_1, beta_2=beta_2, tau=tau
        )

    def step(self, gradients, net, lr, f, byz):
        if len(gradients) < 2:
            raise ValueError(
                "FedAdam requires at least one worker gradient and one server/root gradient"
            )

        param_list = _flatten_round_gradients(gradients)
        param_list = byz(param_list, net, lr, f)
        num_participants = len(param_list) - 1
        if num_participants <= 0:
            raise ValueError("no worker updates available for FedAdam")

        stacked = nd.concat(*[param_list[i] for i in range(num_participants)], dim=1)
        mean_update = nd.mean(stacked, axis=1, keepdims=True)
        adaptive_update = self.core.step(mean_update)
        _apply_model_update(net, adaptive_update, 1.0)

    def state_dict(self):
        core_state = self.core.state_dict()
        if core_state is None:
            return None
        state = dict(core_state)
        state["aggregator_kind"] = np.asarray(["flat_fedadam"], dtype=object)
        state["num_workers"] = np.asarray([int(self.num_workers)], dtype=np.int64)
        return state

    def load_state_dict(self, state, ctx):
        if str(state["aggregator_kind"][0]) != "flat_fedadam":
            raise ValueError("FedAdamAggregator state kind mismatch")
        if int(state["num_workers"][0]) != self.num_workers:
            raise ValueError(
                "FedAdamAggregator worker-count mismatch: expected %d, got %d"
                % (self.num_workers, int(state["num_workers"][0]))
            )
        core_state = dict(state)
        core_state["aggregator_kind"] = np.asarray(["fedadam_core"], dtype=object)
        self.core.load_state_dict(core_state, ctx)


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
        return _flatten_round_gradients(gradients)

    @staticmethod
    def _apply_model_update(net, update_vector, lr):
        _apply_model_update(net, update_vector, lr)

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

    def state_dict(self):
        if self.c_global is None or self.c_local is None:
            return None
        active_c_local = [_nd_to_numpy(value) for value in self.c_local]
        if len(active_c_local) == 0:
            return None
        return {
            "aggregator_kind": np.asarray(["flat_scaffold"], dtype=object),
            "tensor_shape": np.asarray(self.c_global.shape, dtype=np.int64),
            "num_workers": np.asarray([int(self.num_workers)], dtype=np.int64),
            "total_clients": np.asarray([int(self.total_clients)], dtype=np.int64),
            "c_global": _nd_to_numpy(self.c_global),
            "c_local_stack": np.stack(active_c_local, axis=0),
        }

    def load_state_dict(self, state, ctx):
        if str(state["aggregator_kind"][0]) != "flat_scaffold":
            raise ValueError("ScaffoldAggregator state kind mismatch")
        if int(state["num_workers"][0]) != self.num_workers:
            raise ValueError(
                "ScaffoldAggregator worker-count mismatch: expected %d, got %d"
                % (self.num_workers, int(state["num_workers"][0]))
            )
        if int(state["total_clients"][0]) != self.total_clients:
            raise ValueError(
                "ScaffoldAggregator total-client mismatch: expected %d, got %d"
                % (self.total_clients, int(state["total_clients"][0]))
            )
        saved_shape = tuple(int(x) for x in state["tensor_shape"].tolist())
        c_global = _numpy_to_nd(state["c_global"], ctx)
        c_local_stack = state["c_local_stack"]
        if tuple(c_global.shape) != saved_shape:
            raise ValueError("ScaffoldAggregator c_global shape mismatch")
        if c_local_stack.shape[0] != self.num_workers:
            raise ValueError("ScaffoldAggregator c_local length mismatch")
        self.c_global = c_global
        self.c_local = []
        for worker_idx in range(c_local_stack.shape[0]):
            local_vec = _numpy_to_nd(c_local_stack[worker_idx], ctx)
            if tuple(local_vec.shape) != saved_shape:
                raise ValueError(
                    "ScaffoldAggregator c_local[%d] shape mismatch" % (worker_idx,)
                )
            self.c_local.append(local_vec)
