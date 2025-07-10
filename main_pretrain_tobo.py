import os
import time
import json
import argparse
import datetime

from pathlib import Path

import numpy as np

import torch
import torch.backends.cudnn as cudnn

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.kinetics_mfmae import MultiKinetics

import models_tobo

from engine_tobo import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser("ToBo pre-training", add_help=False)

    # Training
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./log")

    # Model parameters
    parser.add_argument("--model", default="mae_vit_small_patch16", type=str,
                        metavar="MODEL", help="Name of model to train")

    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--norm_pix_loss", action="store_true",
                        help="Use (per-patch) normalized pixels as targets for computing loss")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--mask_ratio", default=0.75, type=float)

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=None,
                        metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=1.5e-4,
                        metavar="LR", help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=40)

    # Dataset parameters
    parser.add_argument("--data_path", default="/data/kinetics400", type=str)
    parser.add_argument("--max_distance", default=48, type=int)
    parser.add_argument("--repeated_sampling", type=int, default=2)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--prefetch_factor", default=2, type=int)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    # customize
    parser.add_argument("--framewise_flip", action="store_true")
    parser.add_argument("--num_frames", default=4, type=int)
    parser.add_argument("--exclude_self_frame", action="store_true")

    parser.add_argument("--mask_ratio_src", default=0.0, type=float)

    # Hyperparameters for ADT
    parser.add_argument("--min_frames", default=1, type=int)
    parser.add_argument("--max_frames", default=3, type=int)
    parser.add_argument("--tgt_path", default="csm", type=str)
    parser.add_argument("--tobo_path", default="sm", type=str)

    parser.add_argument("--w_mim", type=float, default=1.0)
    parser.add_argument("--w_tobo", type=float, default=1.0)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = MultiKinetics(
        args.data_path,
        max_distance=args.max_distance,
        repeated_sampling=args.repeated_sampling,
        num_frames=args.num_frames,
        framewise_flip=args.framewise_flip,
    )

    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and not args.eval:
        if args.log_dir.lower() == 'none':
            log_writer = None
        elif args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = logger.TensorboardLogger(log_dir=args.log_dir)
        else:
            log_writer = None
    else:
        log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=args.prefetch_factor,
        drop_last=True,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )

    # define the model
    if "tobo" in args.model:
        model = models_tobo.__dict__[args.model](
            norm_pix_loss=args.norm_pix_loss,
            mask_ratio=args.mask_ratio,
            batch_size=args.batch_size,
            repeated_sampling=args.repeated_sampling,
            num_frames=args.num_frames,
            mask_ratio_src=args.mask_ratio_src,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            tgt_path=args.tgt_path,
            tobo_path=args.tobo_path,
        )

    model.to(device)


    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    args.epochs = args.epochs // args.repeated_sampling
    args.warmup_epochs = args.warmup_epochs // args.repeated_sampling

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None and log_writer.logger_type() == 'tensorboard':
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % 50 == 0 or epoch in [args.epochs - 2, args.epochs - 1, args.epochs]):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=args.repeated_sampling * (epoch+1),
            )
        else:
            output_dir = Path(args.output_dir)
            epoch_name = str(epoch)
            checkpoint_paths = [output_dir / ("checkpoint-last.pth")]
            for checkpoint_path in checkpoint_paths:
                to_save = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "scaler": loss_scaler.state_dict(),
                    "args": args,
                }
                misc.save_on_master(to_save, checkpoint_path)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        # Logger
        if log_writer.logger_type() == 'tensorboard':
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
