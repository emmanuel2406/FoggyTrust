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

# TODO: implement FedAvg (ER), Krum (TG), Trimmed-mean (ER), Median (ER)
