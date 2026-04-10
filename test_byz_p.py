from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import argparse
import byzantine
import os
import sys

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
        choices=("fltrust", "fedavg", "trimmed_mean", "krum"),
    )
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)

def get_device(device):
    # define the device to use
    if device == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(device)
    return ctx
    
def get_cnn(num_outputs=10):
    # define the architecture of the CNN
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(100, activation="relu"))
        cnn.add(gluon.nn.Dense(num_outputs))
    return cnn

def get_net(net_type, num_outputs=10):
    # define the model architecture
    if net_type == 'cnn':
        net = get_cnn(num_outputs)
    else:
        raise NotImplementedError
    return net
    
def get_shapes(dataset):
    # determine the input/output shapes 
    if dataset == 'FashionMNIST' or dataset == 'mnist':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
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

AGGREGATION_FUNCS = {
    "fltrust": nd_aggregation.fltrust,
    "fedavg": nd_aggregation.fedavg,
    "trimmed_mean": nd_aggregation.trimmed_mean,
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
    elif byz_type == 'scale_attack':
        return byzantine.scale_attack
    elif byz_type == 'krum_attack':
        return byzantine.krum_attack
    elif byz_type == 'adaptive_attack':
        return byzantine.adaptive_attack
    else:
        raise NotImplementedError
        
def load_data(dataset):
    # load the dataset
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'mnist':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000, shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 256, shuffle=False, last_batch='rollover')


    else:
        raise NotImplementedError
    return train_data, test_data
    
def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST", seed=1):
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_per_group = num_workers / num_labels

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
            if dataset == "FashionMNIST" or dataset == "mnist":
                x = x.as_in_context(ctx).reshape(1,1,28,28)
            else:
                raise NotImplementedError
            y = y.as_in_context(ctx)
            
            upper_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()
            
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()
            
            if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y.asnumpy())] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
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
    num_inputs, num_outputs, num_labels = get_shapes(args.dataset)
    byz = get_byz(args.byz_type)
    aggregate = get_aggregation(args.aggregation)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter

    lr = lr/batch_size
    
    paraString = 'p'+str(args.p)+ '_' + str(args.dataset) + "server " + str(args.server_pc) + "bias" + str(args.bias)+ "+nworkers " + str(
        args.nworkers) + "+" + "net " + str(args.net) + "+" + "niter " + str(args.niter) + "+" + "lr " + str(
        args.lr) + "+" + "batch_size " + str(args.batch_size) + "+nbyz " + str(
        args.nbyz) + "+" + "byz_type " + str(args.byz_type) + "+" + "aggregation " + str(args.aggregation) + ".txt"
 
    with ctx:
    
        # model architecture
        net = get_net(args.net, num_outputs)
        # initialization
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        # loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        grad_list = []
        test_acc_list = []
        eval_iteration = []

        # load the data
        # fix the seeds for loading data
        seed = args.nrepeats
        if seed > 0:
            mx.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        train_data, test_data = load_data(args.dataset)
        
        # assign data to the server and clients
        server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers, 
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)


        # begin training
        for e in range(niter):
            for i in range(num_workers):
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    batch_label = each_worker_label[i][minibatch]
                    if args.byz_type == "label_flipping_attack" and i < args.nbyz:
                        batch_label = byzantine.flipped_labels_fltrust(batch_label, num_labels)
                    loss = softmax_cross_entropy(output, batch_label)

                loss.backward()

                grad_list.append([param.grad().copy() for param in net.collect_params().values()])

            # nd_aggregation.* expect len(grad_list) == n_workers + 1; the last slot is the
            # server reference update (used by fltrust; trailing row is ignored by fedavg / trimmed_mean).
            # Same RNG draw as the original fltrust path (indices not used for server forward).
            _ = np.random.choice(
                list(range(server_data.shape[0])), size=args.server_pc, replace=False
            )
            with autograd.record():
                output = net(server_data)
                loss = softmax_cross_entropy(output, server_label)
            loss.backward()
            grad_list.append([param.grad().copy() for param in net.collect_params().values()])
            aggregate(grad_list, net, lr, args.nbyz, byz)

            del grad_list
            grad_list = []
            
            # evaluate the model accuracy
            if (e + 1) % 10 == 0:
                test_accuracy = evaluate_accuracy(test_data, net, ctx)
                eval_iteration.append(e + 1)
                test_acc_list.append(test_accuracy)
                print("[%s - %s] Iteration %02d. Test_acc %0.4f" % (args.aggregation, args.byz_type, e, test_accuracy))

        return {
            "eval_iteration": np.asarray(eval_iteration, dtype=np.int64),
            "test_accuracy": np.asarray(test_acc_list, dtype=np.float64),
        }

if __name__ == "__main__":
    args = parse_args()
    input_str = ' '.join(sys.argv)
    print(input_str)
    _ = main(args)
