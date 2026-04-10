import numpy as np
from mxnet import nd, autograd, gluon

_EPS = 1e-12
_ADAPTIVE_SIGMA = np.sqrt(0.5)
_ADAPTIVE_ETA = 0.01
_ADAPTIVE_GAMMA = 0.005
_ADAPTIVE_Q = 10
_ADAPTIVE_V = 10


def no_byz(v, net, lr, f):
    return v


def _as_numpy_vector(sample):
    return sample.asnumpy().reshape(-1).astype(np.float64, copy=False)


def _safe_unit(vec):
    norm = np.linalg.norm(vec)
    if norm <= _EPS:
        return np.zeros_like(vec)
    return vec / norm


def _fltrust_global_update_from_units(client_units, server_unit, server_norm):
    weighted_sum = np.zeros_like(server_unit)
    total_weight = 0.0
    for unit in client_units:
        trust = max(float(np.dot(unit, server_unit)), 0.0)
        weighted_sum += trust * unit
        total_weight += trust
    if total_weight <= _EPS:
        return np.zeros_like(server_unit)
    return server_norm * weighted_sum / total_weight


def _set_vector(sample, flat_vector):
    reshaped = flat_vector.reshape(sample.shape)
    return nd.array(reshaped, ctx=sample.context).astype(sample.dtype)


def _krum_scores(v, f):
    size = len(v)
    neighbor_count = size - f - 2
    if neighbor_count <= 0:
        raise ValueError(
            "Krum requires at least f + 3 client updates; got n=%d, f=%d"
            % (size, f)
        )
    flattened = [sample.asnumpy().reshape(-1) for sample in v]
    metric = []
    for idx, sample in enumerate(flattened):
        distances = [
            np.linalg.norm(sample - other)
            for j, other in enumerate(flattened)
            if j != idx
        ]
        metric.append(np.sum(np.sort(distances)[:neighbor_count]))
    return metric


def _krum_select_index(v, f):
    metric = _krum_scores(v, f)
    return int(np.argmin(metric))

def trim_attack(v, net, lr, f):
    # local model poisoning attack against Trimmed-mean
    vi_shape = v[0].shape
    v_tran = nd.concat(*v, dim=1)
    maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
    minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
    direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    # let the malicious clients (first f clients) perform the attack
    for i in range(f):
        random_12 = 1. + nd.random.uniform(shape=vi_shape)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v        

def flipped_labels_fltrust(labels, num_classes):
    """
    Symmetric label flip l -> M - l - 1 for M classes (FLTrust paper, same setting as [15]).
    Malicious clients minimize loss against these targets instead of true labels.
    """
    return (num_classes - 1) - labels


def label_flipping_attack(v, net, lr, f):
    """
    Label flipping is a data-poisoning attack: Byzantine workers use flipped_labels_fltrust
    during local training (see test_byz_p). Their submitted gradients are already biased;
    aggregation does not modify them (contrast trim_attack).
    """
    return v



def scale_attack(v, net, lr, f):
    """
    Scaling / model-replacement style attack (Bagdasaryan et al.; see
    scrap/backdoor_federated_learning/training.py). After local training, adversaries
    replace weights with L = G + clip_rate * (X - G) (code comment: "100*X-99*G" when
    clip_rate is large). Here updates are gradients; G is proxied by the server/root
    gradient v[-1], X by the malicious client's locally computed gradient v[i].
    clip_rate = scale_weights / f matches the reference implementation's division by
    the number of colluding adversaries.
    """
    if f <= 0:
        return v
    baseline = v[-1]
    scale_weights = 100.0
    clip_rate = scale_weights / float(f)
    for i in range(f):
        v[i] = baseline + clip_rate * (v[i] - baseline)
    return v


def krum_attack(v, net, lr, f, lower_bound=1e-8, upper_bound=1.0):
    """
    Krum-targeted attack adapted to this repo's flattened update format.
    The first ``f`` client updates are replaced with a shared vector whose
    magnitude is tuned until Krum selects one of the Byzantine clients.
    """
    if f <= 0:
        return v

    client_count = len(v) - 1
    if client_count <= 0:
        return v

    honest_updates = v[f:client_count]
    if len(honest_updates) == 0:
        return v

    try:
        direction = nd.sign(nd.sum(nd.concat(*honest_updates, dim=1), axis=-1, keepdims=True))
        lambda1 = upper_bound
        chosen_index = None

        while True:
            krum_local = [v[i] for i in range(client_count)]
            attack_update = -lambda1 * direction
            for i in range(min(f, client_count)):
                krum_local[i] = attack_update

            chosen_index = _krum_select_index(krum_local, f)
            if chosen_index < f:
                break
            if lambda1 < lower_bound:
                break
            lambda1 /= 2.0

        for i in range(min(f, client_count)):
            v[i] = -lambda1 * direction
    except ValueError:
        return v

    return v


def adaptive_attack(v, net, lr, f):
    """
    Adaptive attack specialized to FLTrust's aggregation rule.
    This follows Algorithm 3: compute e0, ei, ci; initialize e'i with
    Trim attack; run coordinate-wise projected zeroth-order ascent; then
    set g'i = ||g0|| e'i for the malicious clients.
    """
    if f <= 0:
        return v

    client_count = len(v) - 1
    malicious_count = min(f, client_count)
    if malicious_count <= 0 or client_count <= 0:
        return v

    server_vec = _as_numpy_vector(v[-1])
    server_norm = np.linalg.norm(server_vec)
    if server_norm <= _EPS:
        return v
    e0 = server_vec / server_norm

    client_updates = [_as_numpy_vector(v[i]) for i in range(client_count)]
    ei = [_safe_unit(update) for update in client_updates]
    ci = [float(np.dot(unit, e0)) for unit in ei]

    relu_ci = np.maximum(np.asarray(ci, dtype=np.float64), 0.0)
    relu_ci_sum = float(np.sum(relu_ci))
    if relu_ci_sum <= _EPS:
        return v

    clean_weighted_sum = np.zeros_like(e0)
    for weight, unit in zip(relu_ci, ei):
        clean_weighted_sum += weight * unit
    g = server_norm * clean_weighted_sum / relu_ci_sum
    s = np.sign(g)

    initialized_clients = trim_attack(
        [sample.copy() for sample in v[:client_count]], net, lr, malicious_count
    )
    e0i = []
    for i in range(malicious_count):
        unit = _safe_unit(_as_numpy_vector(initialized_clients[i]))
        if np.linalg.norm(unit) <= _EPS:
            unit = -_safe_unit(s)
        e0i.append(unit)

    honest_ei = ei[malicious_count:]
    honest_relu_ci = relu_ci[malicious_count:]

    def h(current_e0i):
        malicious_scores = np.maximum(
            np.asarray([float(np.dot(unit, e0)) for unit in current_e0i], dtype=np.float64),
            0.0,
        )
        denom = float(np.sum(malicious_scores) + np.sum(honest_relu_ci))
        if denom <= _EPS:
            attacked = np.zeros_like(e0)
        else:
            attacked_weighted_sum = np.zeros_like(e0)
            for score, unit in zip(malicious_scores, current_e0i):
                attacked_weighted_sum += score * unit
            for score, unit in zip(honest_relu_ci, honest_ei):
                attacked_weighted_sum += score * unit
            attacked = server_norm * attacked_weighted_sum / denom
        return float(np.dot(s, g - attacked))

    for _ in range(_ADAPTIVE_V):
        for i in range(malicious_count):
            for _ in range(_ADAPTIVE_Q):
                base_value = h(e0i)
                perturbation = np.random.normal(
                    loc=0.0, scale=_ADAPTIVE_SIGMA, size=e0.shape
                )
                trial_e0i = [unit.copy() for unit in e0i]
                trial_e0i[i] = trial_e0i[i] + _ADAPTIVE_GAMMA * perturbation
                trial_value = h(trial_e0i)
                grad_estimate = (
                    (trial_value - base_value) / _ADAPTIVE_GAMMA
                ) * perturbation
                e0i[i] = _safe_unit(
                    e0i[i] + _ADAPTIVE_ETA * grad_estimate
                )

    for i in range(malicious_count):
        poisoned = server_norm * e0i[i]
        v[i] = _set_vector(v[i], poisoned)

    return v

# TODO: implement Label-flipping (ER), Scale Attack (ER), Krum (TG), Trim Attack (TG), Adaptive attack (TG) 
