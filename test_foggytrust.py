from __future__ import print_function

import numpy as np

# Restore deprecated np.bool for MXNet 1.x compatibility with NumPy 1.24+
if not hasattr(np, "bool"):
    np.bool = bool

import random
import sys

import mxnet as mx
from mxnet import autograd, gluon

import byzantine
import foggytrust_aggregation
import foggytrust_data
import test_byz_p as tbp


def build_arg_parser():
    parser = tbp.build_arg_parser()
    aggregation_action = None
    for action in parser._actions:
        if action.dest == "aggregation":
            aggregation_action = action
            break

    if aggregation_action is not None:
        current_choices = tuple(aggregation_action.choices or ())
        if "foggytrust" not in current_choices:
            aggregation_action.choices = current_choices + ("foggytrust",)
        aggregation_action.default = "foggytrust"

    parser.add_argument(
        "--fog_server_pc",
        type=int,
        default=None,
        help="trusted fog-node dataset size per group; defaults to server_pc",
    )
    parser.add_argument(
        "--fog_server_pc_mode",
        type=str,
        default="replicated",
        choices=("replicated", "partitioned"),
        help=(
            "trusted-data layout across fog groups: "
            "'replicated' gives each group fog_server_pc points; "
            "'partitioned' splits fog_server_pc across all groups"
        ),
    )
    parser.add_argument(
        "--foggy_aggregation",
        type=str,
        default="fedavg",
        choices=foggytrust_aggregation.FOGGY_AGGREGATION_CHOICES,
        help=(
            "stage-2 (fog -> cloud) aggregation rule; "
            "stage-1 (workers -> fog) always uses FLTrust"
        ),
    )
    parser.add_argument(
        "--fog_num_groups",
        type=int,
        default=None,
        help="number of fog groups; defaults to num_labels when unset",
    )
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)


def _zero_param_grads(net):
    for param in net.collect_params().values():
        if getattr(param, "grad_req", "write") == "null":
            continue
        if hasattr(param, "zero_grad"):
            param.zero_grad()
            continue
        grad = param.grad()
        if grad is None:
            continue
        if isinstance(grad, (list, tuple)):
            for grad_arr in grad:
                grad_arr[:] = 0
        else:
            grad[:] = 0


def _snapshot_gradients(net, data, label, loss_fn):
    # FoggyTrust needs per-worker and per-fog-root updates from the same global model state,
    # so we snapshot gradients without applying them immediately.
    _zero_param_grads(net)
    with autograd.record():
        output = net(data)
        loss = loss_fn(output, label)
    loss.backward()
    return [param.grad().copy() for param in net.collect_params().values()]


def _sample_worker_minibatch(data, label, batch_size):
    worker_size = int(data.shape[0])
    if worker_size <= 0:
        raise ValueError("worker minibatch source is empty")
    replace = worker_size < batch_size
    batch_indices = np.random.choice(worker_size, size=batch_size, replace=replace)
    return data[batch_indices], label[batch_indices]


def _order_group_gradients(worker_ids, worker_gradients, worker_is_byzantine):
    malicious_gradients = []
    honest_gradients = []
    # Preserve compatibility with the existing attack helpers, which assume malicious updates come first.
    for worker_id in worker_ids:
        if worker_is_byzantine[worker_id]:
            malicious_gradients.append(worker_gradients[worker_id])
        else:
            honest_gradients.append(worker_gradients[worker_id])
    return malicious_gradients + honest_gradients, len(malicious_gradients)


def _print_partition_summary(partition, fog_server_pc, fog_server_pc_mode):
    if fog_server_pc_mode == "partitioned":
        print(
            "FoggyTrust trusted-data mode: partitioned total=%d across %d groups"
            % (fog_server_pc, len(partition.group_workers))
        )
    else:
        print(
            "FoggyTrust trusted-data mode: replicated per_group=%d (%d total across groups)"
            % (fog_server_pc, int(sum(partition.group_trusted_sizes)))
        )
    print(
        "FoggyTrust groups fixed at startup: %d groups"
        % (len(partition.group_workers),)
    )
    for group_id, workers in enumerate(partition.group_workers):
        print(
            "  fog_group=%d workers=%d byzantine=%d trusted_points=%d"
            % (
                group_id,
                len(workers),
                partition.group_byzantine_counts[group_id],
                partition.group_trusted_sizes[group_id],
            )
        )


def main(args):
    ctx = tbp.get_device(args.gpu)
    batch_size = args.batch_size
    byz = tbp.get_byz(args.byz_type)
    lr = args.lr / batch_size
    niter = args.niter
    fog_server_pc = args.server_pc if args.fog_server_pc is None else args.fog_server_pc

    with ctx:
        seed = args.nrepeats
        if seed > 0:
            mx.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        train_data, test_data, dataset_meta = tbp.load_data(args.dataset, args=args)
        _, num_outputs, num_labels = tbp.get_shapes(
            args.dataset,
            snapshot_num_labels=dataset_meta.get("num_labels"),
        )
        net = tbp.get_net(args.net, num_outputs)
        net.collect_params().initialize(
            mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx
        )
        start_iter = tbp._load_checkpoint_if_present(net, args, ctx)
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        test_acc_list = []
        attack_succ_list = []
        eval_iteration = []

        partition = foggytrust_data.build_foggytrust_partition(
            train_data,
            args.bias,
            ctx,
            num_labels=num_labels,
            num_workers=args.nworkers,
            server_pc=args.server_pc,
            p=args.p,
            dataset=args.dataset,
            seed=seed,
            fog_server_pc=args.fog_server_pc,
            fog_server_pc_mode=args.fog_server_pc_mode,
            nbyz=args.nbyz,
            fog_num_groups=args.fog_num_groups,
        )

        _print_partition_summary(partition, fog_server_pc, args.fog_server_pc_mode)
        num_fog_groups = len(partition.group_workers)
        # Estimate fog-level adversarial count as groups containing at least one Byzantine worker.
        fog_level_nbyz = int(
            sum(1 for byz_count in partition.group_byzantine_counts if byz_count > 0)
        )
        fog_stage2_aggregator = foggytrust_aggregation.build_foggy_stage2_aggregator(
            aggregation=args.foggy_aggregation,
            num_fog_nodes=num_fog_groups,
            fog_nbyz=fog_level_nbyz,
            fedadam_eta=args.fedadam_eta,
            fedadam_beta_1=args.fedadam_beta_1,
            fedadam_beta_2=args.fedadam_beta_2,
            fedadam_tau=args.fedadam_tau,
        )

        if args.byz_type == "scaling_attack":
            print(
                "Scaling attack: train with labels %d -> %d on Byzantine clients; "
                "attack success rate = fraction of test images with true label %d "
                "that the global model predicts as %d (FoggyTrust)."
                % (
                    args.scaling_source_label,
                    args.scaling_target_label,
                    args.scaling_source_label,
                    args.scaling_target_label,
                )
            )

        for e in range(start_iter, niter):
            worker_gradients = [None for _ in range(args.nworkers)]

            for worker_id in range(args.nworkers):
                # Every worker computes a local update against the current global model.
                batch_data, batch_label = _sample_worker_minibatch(
                    partition.worker_data[worker_id],
                    partition.worker_label[worker_id],
                    batch_size,
                )

                if partition.worker_is_byzantine[worker_id]:
                    if args.byz_type == "label_flipping_attack":
                        batch_label = byzantine.flipped_labels_fltrust(
                            batch_label, num_labels
                        )
                    elif args.byz_type == "scaling_attack":
                        batch_label = byzantine.scaling_poison_labels(
                            batch_label,
                            args.scaling_source_label,
                            args.scaling_target_label,
                        )

                worker_gradients[worker_id] = _snapshot_gradients(
                    net,
                    batch_data,
                    batch_label,
                    softmax_cross_entropy,
                )

            fog_updates = []
            for group_id, worker_ids in enumerate(partition.group_workers):
                # Stage 1 (clients -> fog): FLTrust-style aggregation.
                # The fog node's trusted mini-dataset produces a root update used as the
                # FLTrust baseline; worker updates are trust-weighted/norm-clipped against it.
                ordered_group_gradients, local_byz_count = _order_group_gradients(
                    worker_ids,
                    worker_gradients,
                    partition.worker_is_byzantine,
                )
                fog_root_gradient = _snapshot_gradients(
                    net,
                    partition.fog_server_data[group_id],
                    partition.fog_server_label[group_id],
                    softmax_cross_entropy,
                )
                fog_update = foggytrust_aggregation.fltrust_group_update(
                    ordered_group_gradients + [fog_root_gradient],
                    net,
                    lr,
                    local_byz_count,
                    byz,
                )
                fog_updates.append(fog_update)

            # Stage 2 (fog -> cloud): choose fog-level aggregation independently.
            global_update = fog_stage2_aggregator.aggregate(fog_updates)
            foggytrust_aggregation.apply_update_vector(net, global_update, lr)

            if (e + 1) % 10 == 0:
                eval_iteration.append(e + 1)
                if args.byz_type == "scaling_attack":
                    test_accuracy, asr = tbp.evaluate_test_accuracy_and_scaling_asr(
                        test_data,
                        net,
                        ctx,
                        args.scaling_source_label,
                        args.scaling_target_label,
                    )
                    test_acc_list.append(test_accuracy)
                    attack_succ_list.append(asr)
                    print(
                        "[foggytrust - %s] Iteration %02d. Test_acc %0.4f  "
                        "Attack_succ_rate %0.4f (src=%d tgt=%d)"
                        % (
                            args.byz_type,
                            e,
                            test_accuracy,
                            asr,
                            args.scaling_source_label,
                            args.scaling_target_label,
                        )
                    )
                else:
                    test_accuracy = tbp.evaluate_accuracy(test_data, net, ctx)
                    test_acc_list.append(test_accuracy)
                    print(
                        "[foggytrust - %s] Iteration %02d. Test_acc %0.4f"
                        % (args.byz_type, e, test_accuracy)
                    )
            tbp._save_checkpoint_if_requested(net, args, e + 1)

        out = {
            "eval_iteration": np.asarray(eval_iteration, dtype=np.int64),
            "test_accuracy": np.asarray(test_acc_list, dtype=np.float64),
        }
        if args.byz_type == "scaling_attack":
            out["attack_success_rate"] = np.asarray(
                attack_succ_list, dtype=np.float64
            )
        else:
            out["attack_success_rate"] = None
        return out


if __name__ == "__main__":
    args = parse_args()
    print(" ".join(sys.argv))
    _ = main(args)
