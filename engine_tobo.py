import sys
import math

from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None and log_writer.logger_type() == 'tensorboard':
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batches in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        multi_samples = batches[0]

        multi_samples = multi_samples.to(device, non_blocking=True)
        multi_samples = multi_samples.reshape(-1, *multi_samples.shape[2:])  # torch.Size([32, 2, 4, 3, 224, 224])
        multi_samples = multi_samples.chunk(args.num_frames, dim=1)

        list_frames = []
        for i in range(len(multi_samples)):
            list_frames.append(multi_samples[i].squeeze())

        with torch.cuda.amp.autocast():
            loss_mim, loss_tobo = model(
                list_imgs=list_frames, epoch=data_iter_step / len(data_loader) + epoch,
                # list_idx=list_indices
            )

        if loss_tobo is None:
            loss = loss_mim
            loss_value = loss.item()
            loss_mim_value = loss_mim.item()
            loss_tobo_value = 0
        else:
            loss = args.w_mim * loss_mim + args.w_tobo * loss_tobo
            loss_value = loss.item()
            loss_mim_value = loss_mim.item()
            loss_tobo_value = loss_tobo.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_mim=loss_mim_value)
        metric_logger.update(loss_tobo=loss_tobo_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0 and log_writer.logger_type() == 'tensorboard':
            log_writer.update(loss=loss_value_reduce, head="train_loss")
            log_writer.update(lr=lr, head="opt")
            log_writer.update(loss_mim=loss_mim_value, head="loss_mim")
            log_writer.update(loss_tobo=loss_tobo_value, head="loss_tobo")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
