import numpy as np

# Restore deprecated np.bool for MXNet 1.x compatibility with NumPy 1.24+
if not hasattr(np, "bool"):
    np.bool = bool

from mxnet import nd
import nd_aggregation

FOGGY_AGGREGATION_CHOICES = (
    "fedavg",
    "trimmed_mean",
    "median",
    "krum",
    "scaffold",
    "fedadam",
)


def flatten_gradients(gradient_list):
    if not gradient_list:
        raise ValueError("gradient_list must not be empty")
    # FLTrust operates on one flattened update vector per participant.
    return nd.concat(*[grad.reshape((-1, 1)) for grad in gradient_list], dim=0)


def fltrust_group_update(gradients, net, lr, f, byz):
    """
    gradients: list of gradient lists. The last gradient list is the fog-node root update.
    net: model parameters (passed through to byzantine handlers for compatibility).
    lr: learning rate (passed through to byzantine handlers for compatibility).
    f: number of malicious workers in this fog group. The first f submitted updates are malicious.
    byz: attack type.
    """
    if len(gradients) < 2:
        raise ValueError(
            "FoggyTrust requires at least one worker update and one fog-root update"
        )

    # The byzantine hooks in this repo expect the first f submitted updates to be malicious.
    param_list = [flatten_gradients(gradient_set).copy() for gradient_set in gradients]
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1

    # The final entry is the fog node's trusted root update, which plays the role of the FLTrust baseline.
    baseline = nd.array(param_list[-1]).squeeze()
    cos_sim = []
    new_param_list = []

    for each_param_list in param_list:
        each_param_array = nd.array(each_param_list).squeeze()
        cos_sim.append(
            nd.dot(baseline, each_param_array)
            / (nd.norm(baseline) + 1e-9)
            / (nd.norm(each_param_array) + 1e-9)
        )

    cos_sim = nd.stack(*cos_sim)[:-1]
    cos_sim = nd.maximum(cos_sim, 0)
    normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9)

    # Reuse FLTrust's trust-weighted, norm-clipped aggregation exactly within each fog group.
    for idx in range(n):
        new_param_list.append(
            param_list[idx]
            * normalized_weights[idx]
            / (nd.norm(param_list[idx]) + 1e-9)
            * nd.norm(baseline)
        )

    if not new_param_list:
        return nd.zeros((baseline.size, 1), ctx=baseline.context, dtype=baseline.dtype)

    return nd.sum(nd.concat(*new_param_list, dim=1), axis=-1, keepdims=True)


def mean_fog_updates(update_vectors):
    if not update_vectors:
        raise ValueError("update_vectors must not be empty")
    # v1 cloud aggregation is an equal mean over fog-node updates.
    # Future weighting / FedProx-style cloud aggregation can replace this helper.
    return nd.mean(nd.concat(*update_vectors, dim=1), axis=1, keepdims=True)


def trimmed_mean_fog_updates(update_vectors, fog_nbyz):
    if not update_vectors:
        raise ValueError("update_vectors must not be empty")
    n = len(update_vectors)
    k = int(max(0, fog_nbyz))
    max_k = max(0, (n - 1) // 2)
    if 2 * k >= n:
        k = max_k
    stacked = nd.concat(*update_vectors, dim=1)
    sorted_vals = nd.sort(stacked, axis=1)
    trimmed = sorted_vals[:, k : n - k]
    return nd.mean(trimmed, axis=1, keepdims=True)


def median_fog_updates(update_vectors):
    if not update_vectors:
        raise ValueError("update_vectors must not be empty")
    n = len(update_vectors)
    stacked = nd.concat(*update_vectors, dim=1)
    sorted_vals = nd.sort(stacked, axis=1)
    mid_lo = (n - 1) // 2
    mid_hi = n // 2
    if mid_lo == mid_hi:
        return sorted_vals[:, mid_lo : mid_lo + 1]
    return (sorted_vals[:, mid_lo : mid_lo + 1] + sorted_vals[:, mid_hi : mid_hi + 1]) / 2


def krum_fog_updates(update_vectors, fog_nbyz):
    if not update_vectors:
        raise ValueError("update_vectors must not be empty")
    n = len(update_vectors)
    f = int(max(0, fog_nbyz))
    neighbor_count = n - f - 2
    if neighbor_count <= 0:
        raise ValueError(
            "Fog-stage Krum requires at least f + 3 fog updates; got n=%d, f=%d"
            % (n, f)
        )
    flattened = [x.asnumpy().reshape(-1) for x in update_vectors]
    scores = []
    for i, sample in enumerate(flattened):
        distances = [
            np.linalg.norm(sample - other)
            for j, other in enumerate(flattened)
            if j != i
        ]
        scores.append(np.sum(np.sort(distances)[:neighbor_count]))
    winner = int(np.argmin(scores))
    return update_vectors[winner].copy()


class ScaffoldFogAggregator(object):
    """
    Stateful SCAFFOLD-style cloud aggregator over fog updates.

    The incoming fog updates are already flattened vectors (d, 1).
    """

    def __init__(self, num_fog_nodes, total_fog_nodes=None):
        self.num_fog_nodes = int(num_fog_nodes)
        self.total_fog_nodes = int(
            total_fog_nodes if total_fog_nodes is not None else num_fog_nodes
        )
        if self.num_fog_nodes <= 0:
            raise ValueError("num_fog_nodes must be positive")
        if self.total_fog_nodes <= 0:
            raise ValueError("total_fog_nodes must be positive")
        self.c_global = None
        self.c_local = None

    def _ensure_state(self, template_vector, num_participants):
        if self.c_global is None:
            self.c_global = nd.zeros_like(template_vector)
        if self.c_local is None:
            self.c_local = [nd.zeros_like(template_vector) for _ in range(self.num_fog_nodes)]
        if len(self.c_local) < num_participants:
            self.c_local.extend(
                [
                    nd.zeros_like(template_vector)
                    for _ in range(num_participants - len(self.c_local))
                ]
            )

    def aggregate(self, update_vectors):
        if not update_vectors:
            raise ValueError("update_vectors must not be empty")
        num_participants = len(update_vectors)
        template = update_vectors[0]
        self._ensure_state(template, num_participants)

        corrected_updates = []
        c_deltas = []
        for fog_idx in range(num_participants):
            fog_update = update_vectors[fog_idx]
            corrected_updates.append(
                fog_update + self.c_global - self.c_local[fog_idx]
            )
            prev_c_local = self.c_local[fog_idx]
            next_c_local = fog_update.copy()
            c_deltas.append(next_c_local - prev_c_local)
            self.c_local[fog_idx] = next_c_local

        global_update = nd.mean(nd.concat(*corrected_updates, dim=1), axis=1, keepdims=True)
        scale = float(num_participants) / float(self.total_fog_nodes)
        c_global_delta = nd.mean(nd.concat(*c_deltas, dim=1), axis=1, keepdims=True)
        self.c_global = self.c_global + scale * c_global_delta
        return global_update


class FedAdamFogAggregator(object):
    """
    Stateful FedAdam cloud aggregator over already-flattened fog updates.

    The incoming fog updates are averaged uniformly before the FedAdam server
    optimizer transforms them into one adaptive cloud update vector.
    """

    def __init__(self, eta=0.1, beta_1=0.9, beta_2=0.99, tau=1e-3):
        self.core = nd_aggregation.FedAdamCore(
            eta=eta, beta_1=beta_1, beta_2=beta_2, tau=tau
        )

    def aggregate(self, update_vectors):
        if not update_vectors:
            raise ValueError("update_vectors must not be empty")
        mean_update = nd.mean(nd.concat(*update_vectors, dim=1), axis=1, keepdims=True)
        return self.core.step(mean_update)


class FoggyStage2Aggregator(object):
    """Selector/factory wrapper for fog -> cloud aggregation."""

    def __init__(
        self,
        aggregation,
        num_fog_nodes,
        fog_nbyz=0,
        fedadam_eta=0.1,
        fedadam_beta_1=0.9,
        fedadam_beta_2=0.99,
        fedadam_tau=1e-3,
    ):
        aggregation_name = str(aggregation).strip().lower()
        if aggregation_name not in FOGGY_AGGREGATION_CHOICES:
            raise NotImplementedError(
                "Unknown foggy_aggregation: %r. Choices: %s"
                % (aggregation, ", ".join(FOGGY_AGGREGATION_CHOICES))
            )
        self.aggregation = aggregation_name
        self.num_fog_nodes = int(num_fog_nodes)
        self.fog_nbyz = int(max(0, fog_nbyz))
        self._scaffold = None
        self._fedadam = None
        if self.aggregation == "scaffold":
            self._scaffold = ScaffoldFogAggregator(
                num_fog_nodes=self.num_fog_nodes, total_fog_nodes=self.num_fog_nodes
            )
        if self.aggregation == "fedadam":
            self._fedadam = FedAdamFogAggregator(
                eta=fedadam_eta,
                beta_1=fedadam_beta_1,
                beta_2=fedadam_beta_2,
                tau=fedadam_tau,
            )

    def aggregate(self, update_vectors):
        if self.aggregation == "fedavg":
            return mean_fog_updates(update_vectors)
        if self.aggregation == "trimmed_mean":
            return trimmed_mean_fog_updates(update_vectors, self.fog_nbyz)
        if self.aggregation == "median":
            return median_fog_updates(update_vectors)
        if self.aggregation == "krum":
            return krum_fog_updates(update_vectors, self.fog_nbyz)
        if self.aggregation == "scaffold":
            return self._scaffold.aggregate(update_vectors)
        if self.aggregation == "fedadam":
            return self._fedadam.aggregate(update_vectors)
        raise NotImplementedError("Unknown foggy_aggregation: %r" % (self.aggregation,))


def build_foggy_stage2_aggregator(
    aggregation,
    num_fog_nodes,
    fog_nbyz=0,
    fedadam_eta=0.1,
    fedadam_beta_1=0.9,
    fedadam_beta_2=0.99,
    fedadam_tau=1e-3,
):
    return FoggyStage2Aggregator(
        aggregation=aggregation,
        num_fog_nodes=num_fog_nodes,
        fog_nbyz=fog_nbyz,
        fedadam_eta=fedadam_eta,
        fedadam_beta_1=fedadam_beta_1,
        fedadam_beta_2=fedadam_beta_2,
        fedadam_tau=fedadam_tau,
    )


def apply_update_vector(net, update_vector, lr):
    flat_update = update_vector.reshape((-1,))
    idx = 0
    # Apply the hierarchical update once so the cloud model is updated exactly one time per round.
    for param in net.collect_params().values():
        next_idx = idx + param.data().size
        param_update = flat_update[idx:next_idx].reshape(param.data().shape)
        param.set_data(param.data() - lr * param_update)
        idx = next_idx
