from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import argparse
import byzantine
import glob
import os
import re
import sys
import safari_helper
import model_helper

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="FashionMNIST")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.006)
    parser.add_argument("--nworkers", help="# workers", type=int, default=100)
    parser.add_argument("--niter", help="# iterations", type=int, default=2500)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
    parser.add_argument("--nrepeats", help="seed", type=int, default=0)
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=20)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no")
    parser.add_argument(
        "--aggregation",
        help="aggregation rule",
        type=str,
        default="fltrust",
        choices=("fltrust", "fedavg", "trimmed_mean", "median", "krum", "scaffold"),
    )
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    parser.add_argument(
        "--scaling_source_label",
        type=int,
        default=7,
        help="true class of attacker-chosen target test points for scaling attack ASR",
    )
    parser.add_argument(
        "--scaling_target_label",
        type=int,
        default=1,
        help="attacker-chosen target label for scaling attack (train + ASR)",
    )
    safari_helper.add_snapshot_safari_args(parser)
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)


_CHECKPOINT_ITER_RE = re.compile(r"__iter_(\d+)$")


def _checkpoint_path_for_iteration(checkpoint_path, iteration):
    base, ext = os.path.splitext(checkpoint_path)
    return "%s__iter_%06d%s" % (base, int(iteration), ext)


def _checkpoint_schedule(niter):
    niter_i = max(1, int(niter))
    return {
        max(1, min(niter_i, int(round(float(niter_i) * (k / 10.0)))))
        for k in range(1, 11)
    }


def _is_checkpoint_iteration(iteration, niter):
    if iteration is None:
        return False
    return int(iteration) in _checkpoint_schedule(niter)


def _find_latest_checkpoint(checkpoint_path, niter):
    base, ext = os.path.splitext(checkpoint_path)
    best_iter = -1
    best_path = None
    pattern = "%s__iter_*%s" % (base, ext)
    for cand in glob.glob(pattern):
        stem = os.path.splitext(cand)[0]
        m = _CHECKPOINT_ITER_RE.search(stem)
        if not m:
            continue
        it = int(m.group(1))
        if niter is not None and it > int(niter):
            continue
        if it > best_iter:
            best_iter = it
            best_path = cand
    return best_path, best_iter


def _load_checkpoint_if_present(net, args, ctx):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        return 0
    latest_path, latest_iter = _find_latest_checkpoint(
        checkpoint_path, getattr(args, "niter", None)
    )
    if latest_path is None:
        if os.path.exists(checkpoint_path):
            # Backward compatibility with legacy single-file checkpoints.
            net.load_parameters(checkpoint_path, ctx=ctx)
            print("Loaded legacy checkpoint:", checkpoint_path)
            return 0
        print("Checkpoint not found, starting fresh:", checkpoint_path)
        return 0
    net.load_parameters(latest_path, ctx=ctx)
    print("Loaded checkpoint:", latest_path)
    return int(latest_iter)


def _save_checkpoint_if_requested(net, args, iteration):
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if not checkpoint_path:
        return
    if not _is_checkpoint_iteration(iteration, getattr(args, "niter", 1)):
        return
    iter_path = _checkpoint_path_for_iteration(checkpoint_path, iteration)
    checkpoint_dir = os.path.dirname(os.path.abspath(iter_path))
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    net.save_parameters(iter_path)
    print("Saved checkpoint:", iter_path)

def get_device(device):
    # define the device to use
    if device == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(device)
    return ctx
    
def _reshape_sample(x, ctx):
    x_ctx = x.as_in_context(ctx)
    if len(x_ctx.shape) == 2:
        return x_ctx.reshape(1, 1, int(x_ctx.shape[0]), int(x_ctx.shape[1]))
    if len(x_ctx.shape) == 3:
        return x_ctx.reshape(1, int(x_ctx.shape[0]), int(x_ctx.shape[1]), int(x_ctx.shape[2]))
    if len(x_ctx.shape) == 4:
        return x_ctx
    raise ValueError("Unsupported sample shape: %r" % (tuple(x_ctx.shape),))


def get_shapes(dataset, snapshot_num_labels=None):
    # determine the input/output shapes 
    if dataset == 'FashionMNIST' or dataset == 'mnist':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    elif dataset == "SnapshotSafari":
        if snapshot_num_labels is None:
            raise ValueError("snapshot_num_labels is required for dataset SnapshotSafari")
        num_inputs = None
        num_outputs = int(snapshot_num_labels)
        num_labels = int(snapshot_num_labels)
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels

def evaluate_accuracy(data_iterator, net, ctx, trigger=False, target=None):
    # evaluate the (attack) accuracy of the model
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        # if trigger:
        #     data, label, remaining_idx, add_backdoor(data, label, trigger, target)
        output = net(data)
        predictions = nd.argmax(output, axis=1)                
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
        acc.update(preds=predictions, labels=label)        
    return acc.get()[1]


def evaluate_test_accuracy_and_scaling_asr(
    data_iterator, net, ctx, asr_source_label, asr_target_label
):
    """
    One full pass over the test loader: overall accuracy (mx.metric) plus scaling
    attack success rate. Using a single pass avoids Gluon DataLoader quirks where
    a second ``for`` over the same loader can misbehave, which produced unstable ASR.
    ASR is in [0, 1] (a fraction; 1.0 means 100% of source-class test points predict
    the target class, not an overflow).
    """
    acc = mx.metric.Accuracy()
    src_int = int(asr_source_label)
    tgt_int = int(asr_target_label)
    total = 0
    success = 0
    for _, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        lb = np.rint(label.asnumpy().reshape(-1)).astype(np.int64)
        pr = predictions.astype("int64").asnumpy().reshape(-1)
        sel = lb == src_int
        total += int(np.sum(sel))
        success += int(np.sum(sel & (pr == tgt_int)))
    acc_val = acc.get()[1]
    if total <= 0:
        return acc_val, float("nan")
    rate = float(success) / float(total)
    # Should hold exactly; clamp only against float noise if any downstream assumes [0,1]
    return acc_val, min(1.0, max(0.0, rate))


def evaluate_scaling_attack_success_rate(
    data_iterator, net, ctx, source_label, target_label
):
    """
    Attack success rate (FLTrust): among test examples whose true label is
    ``source_label``, the fraction whose predicted class is ``target_label``.
    """
    _, asr = evaluate_test_accuracy_and_scaling_asr(
        data_iterator, net, ctx, source_label, target_label
    )
    return asr

AGGREGATION_FUNCS = {
    "fltrust": nd_aggregation.fltrust,
    "fedavg": nd_aggregation.fedavg,
    "trimmed_mean": nd_aggregation.trimmed_mean,
    "median": nd_aggregation.median,
    "krum": nd_aggregation.krum,
}


def get_aggregation(aggregation):
    if aggregation not in AGGREGATION_FUNCS:
        raise NotImplementedError("Unknown aggregation: %r" % (aggregation,))
    return AGGREGATION_FUNCS[aggregation]


def get_byz(byz_type):
    # get the attack type
    if byz_type == "no":
        return byzantine.no_byz
    elif byz_type == 'trim_attack':
        return byzantine.trim_attack
    elif byz_type == 'label_flipping_attack' :
        return byzantine.label_flipping_attack
    elif byz_type == 'krum_attack':
        return byzantine.krum_attack
    elif byz_type == 'scaling_attack':
        return byzantine.scaling_attack
    elif byz_type == 'adaptive_attack':
        return byzantine.adaptive_attack
    else:
        raise NotImplementedError
        
def load_data(dataset, args=None):
    # load the dataset
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
        meta = {"num_labels": 10}
    elif dataset == 'mnist':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000, shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 256, shuffle=False, last_batch='rollover')
        meta = {"num_labels": 10}
    elif dataset == "SnapshotSafari":
        if args is None:
            raise ValueError("args are required when dataset is SnapshotSafari")
        train_data, test_data, meta = safari_helper.load_snapshot_safari_data(
            args, batch_size=256, last_batch="rollover"
        )

    else:
        raise NotImplementedError
    return train_data, test_data, meta
    
def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST", seed=1):
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_counts = [
        (num_workers // num_labels) + (1 if group_id < (num_workers % num_labels) else 0)
        for group_id in range(num_labels)
    ]
    group_workers = []
    worker_cursor = 0
    for count in worker_counts:
        group_workers.append(list(range(worker_cursor, worker_cursor + count)))
        worker_cursor += count

    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]   
    server_data = []
    server_label = [] 
    
    # compute the labels needed for each class
    real_dis = [1. / num_labels for _ in range(num_labels)]
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

    # randomly assign the data points based on the labels
    server_counter = [0 for _ in range(num_labels)]
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            x = _reshape_sample(x, ctx)
            y = y.as_in_context(ctx)
            label_id = int(np.rint(y.asnumpy()).item())
            
            upper_bound = label_id * (1. - bias) / (num_labels - 1) + bias
            lower_bound = label_id * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()
            
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + label_id + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = label_id
            worker_group = int(worker_group) % num_labels
            
            if server_counter[label_id] < samp_dis[label_id]:
                server_data.append(x)
                server_label.append(y)
                server_counter[label_id] += 1
            else:
                rd = np.random.random_sample()
                workers = group_workers[worker_group]
                if not workers:
                    workers = list(range(num_workers))
                selected_worker = workers[int(np.floor(rd * len(workers))) % len(workers)]
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)
                
    server_data = nd.concat(*server_data, dim=0)
    server_label = nd.concat(*server_label, dim=0)
    
    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]
    

    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    
    
    return server_data, server_label, each_worker_data, each_worker_label
    
        
def main(args):
    # device to use
    ctx = get_device(args.gpu)
    batch_size = args.batch_size
    byz = get_byz(args.byz_type)
    num_workers = args.nworkers
    if args.aggregation == "scaffold":
        aggregate = None
        scaffold_aggregator = nd_aggregation.ScaffoldAggregator(
            num_workers=num_workers, total_clients=num_workers
        )
    else:
        aggregate = get_aggregation(args.aggregation)
        scaffold_aggregator = None
    lr = args.lr
    niter = args.niter

    lr = lr/batch_size
    
    paraString = 'p'+str(args.p)+ '_' + str(args.dataset) + "server " + str(args.server_pc) + "bias" + str(args.bias)+ "+nworkers " + str(
        args.nworkers) + "+" + "net " + str(args.net) + "+" + "niter " + str(args.niter) + "+" + "lr " + str(
        args.lr) + "+" + "batch_size " + str(args.batch_size) + "+nbyz " + str(
        args.nbyz) + "+" + "byz_type " + str(args.byz_type) + "+" + "aggregation " + str(args.aggregation) + ".txt"
 
    with ctx:
        # load the data
        # fix the seeds for loading data
        seed = args.nrepeats
        if seed > 0:
            mx.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        train_data, test_data, dataset_meta = load_data(args.dataset, args=args)
        _, num_outputs, num_labels = get_shapes(
            args.dataset,
            snapshot_num_labels=dataset_meta.get("num_labels"),
        )

        # model architecture
        net = model_helper.build_model(args.dataset, args.net, num_outputs)
        # initialization
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        start_iter = _load_checkpoint_if_present(net, args, ctx)
        # loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        grad_list = []
        test_acc_list = []
        attack_succ_list = []
        eval_iteration = []
        
        # assign data to the server and clients
        server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers, 
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)


        if args.byz_type == "scaling_attack":
            print(
                "Scaling attack: train with labels %d -> %d on Byzantine clients; "
                "attack success rate = fraction of test images with true label %d "
                "that the global model predicts as %d (FLTrust)."
                % (
                    args.scaling_source_label,
                    args.scaling_target_label,
                    args.scaling_source_label,
                    args.scaling_target_label,
                )
            )

        # begin training
        for e in range(start_iter, niter):
            for i in range(num_workers):
                local_size = int(each_worker_data[i].shape[0])
                minibatch = np.random.choice(
                    list(range(local_size)),
                    size=batch_size,
                    replace=(local_size < batch_size),
                )
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    batch_label = each_worker_label[i][minibatch]
                    if args.byz_type == "label_flipping_attack" and i < args.nbyz:
                        batch_label = byzantine.flipped_labels_fltrust(batch_label, num_labels)
                    elif args.byz_type == "scaling_attack" and i < args.nbyz:
                        batch_label = byzantine.scaling_poison_labels(
                            batch_label,
                            args.scaling_source_label,
                            args.scaling_target_label,
                        )
                    loss = softmax_cross_entropy(output, batch_label)

                loss.backward()

                grad_list.append([param.grad().copy() for param in net.collect_params().values()])

            # nd_aggregation.* expect len(grad_list) == n_workers + 1; the last slot is the
            # server reference update (used by fltrust; trailing row is ignored by fedavg / trimmed_mean).
            # Same RNG draw as the original fltrust path (indices not used for server forward).
            server_size = int(server_data.shape[0])
            _ = np.random.choice(
                list(range(server_size)),
                size=args.server_pc,
                replace=(server_size < args.server_pc),
            )
            with autograd.record():
                output = net(server_data)
                loss = softmax_cross_entropy(output, server_label)
            loss.backward()
            grad_list.append([param.grad().copy() for param in net.collect_params().values()])
            if args.aggregation == "scaffold":
                scaffold_aggregator.step(grad_list, net, lr, args.nbyz, byz)
            else:
                aggregate(grad_list, net, lr, args.nbyz, byz)

            del grad_list
            grad_list = []
            
            # evaluate the model accuracy
            if (e + 1) % 10 == 0:
                eval_iteration.append(e + 1)
                if args.byz_type == "scaling_attack":
                    test_accuracy, asr = evaluate_test_accuracy_and_scaling_asr(
                        test_data,
                        net,
                        ctx,
                        args.scaling_source_label,
                        args.scaling_target_label,
                    )
                    test_acc_list.append(test_accuracy)
                    attack_succ_list.append(asr)
                    print(
                        "[%s - %s] Iteration %02d. Test_acc %0.4f  Attack_succ_rate %0.4f (src=%d tgt=%d)"
                        % (
                            args.aggregation,
                            args.byz_type,
                            e,
                            test_accuracy,
                            asr,
                            args.scaling_source_label,
                            args.scaling_target_label,
                        )
                    )
                else:
                    test_accuracy = evaluate_accuracy(test_data, net, ctx)
                    test_acc_list.append(test_accuracy)
                    print("[%s - %s] Iteration %02d. Test_acc %0.4f" % (args.aggregation, args.byz_type, e, test_accuracy))
            _save_checkpoint_if_requested(net, args, e + 1)

        out = {
            "eval_iteration": np.asarray(eval_iteration, dtype=np.int64),
            "test_accuracy": np.asarray(test_acc_list, dtype=np.float64),
        }
        if args.byz_type == "scaling_attack":
            out["attack_success_rate"] = np.asarray(attack_succ_list, dtype=np.float64)
        else:
            out["attack_success_rate"] = None
        return out

if __name__ == "__main__":
    args = parse_args()
    input_str = ' '.join(sys.argv)
    print(input_str)
    _ = main(args)
