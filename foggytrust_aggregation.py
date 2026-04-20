import numpy as np

# Restore deprecated np.bool for MXNet 1.x compatibility with NumPy 1.24+
if not hasattr(np, "bool"):
    np.bool = bool

from mxnet import nd


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


def apply_update_vector(net, update_vector, lr):
    flat_update = update_vector.reshape((-1,))
    idx = 0
    # Apply the hierarchical update once so the cloud model is updated exactly one time per round.
    for param in net.collect_params().values():
        next_idx = idx + param.data().size
        param_update = flat_update[idx:next_idx].reshape(param.data().shape)
        param.set_data(param.data() - lr * param_update)
        idx = next_idx
