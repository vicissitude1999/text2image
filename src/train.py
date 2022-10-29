import argparse
import glob
import time
import json
import logging
import os
import sys
import shutil
import random
from random import randint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from addict import Dict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from dataset import MNISTCaptions, Captions
from model import AlignDraw
import utils


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def parse_args():
    config = sys.argv[1]  # "tools/mnist-captions.json

    with open(config) as f:
        args = json.load(f)
    args = Dict(args)

    return args


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, 0.01)
        nn.init.constant_(m.bias.data, 0)


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)
    args = parse_args()
    print(args)
    init_seeds(args.seed, False)
    device = "cuda"

    # output directory
    if args.save:
        args.savedir = "{}/{}".format(args.savedir, time.strftime("%Y%m%d-%H%M%S"))
        utils.create_exp_dir(args.savedir, scripts_to_save=glob.glob("*.py"))
        with open(Path(args.savedir, "args.json"), "w") as f:
            json.dump(args, f, indent=4)
    # logging
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    if args.save:
        fh = logging.FileHandler(Path(args.savedir, "log.txt"), "w")
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        # tensorboard
        writer = SummaryWriter(args.savedir)

    # create dataset
    banned = [randint(0, 10) for i in range(12)]
    train_data = MNISTCaptions(datadir=args.datadir, banned=banned, size=10000, train=True)
    train_queue = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # model = AlignDraw
    model = AlignDraw(
        args.model[0].runSteps,
        args.model[0].dimReadAttent,
        args.model[0].dimWriteAttent,
        args.model[0].dimX,
        args.model[0].dimY,
        args.model[0].dimLangRNN,
        args.model[0].dimRNNEnc,
        args.model[0].dimZ,
        args.model[0].dimRNNDec,
        args.model[0].dimAlign,
        args.channels,
        device,
    ).to(device)
    model.apply(initialize_weights)

    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.reduceLRAfter], gamma=0.1
    )

    best_obj = float("inf")
    for epoch in range(args.epochs):
        logging.info(f"[epoch] {epoch:d}")

        # train
        objs = utils.AvgrageMeter()
        objs_Lx = utils.AvgrageMeter()
        objs_Lz = utils.AvgrageMeter()
        model.train()

        for step, data in enumerate(train_queue):
            image, caption = data
            image = image.to(device, non_blocking=True)
            caption = caption.to(device, non_blocking=True)

            optimizer.zero_grad()
            Lx, Lz = model.loss((image, caption))
            loss = Lx + Lz
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10) # clip norm before step!
            optimizer.step()

            n = caption.size(0)
            objs.update(loss.item(), n)
            objs_Lx.update(Lx.item(), n)
            objs_Lz.update(Lz.item(), n)

            if step % args.report_freq == 0:
                logging.info(f"train {step:03d} loss {objs.avg:.3f} Lx {objs_Lx.avg:.3f} Lz {objs_Lz.avg:.3f}")
                if args.save:
                    writer.add_scalar(
                        "LossBatch", objs.avg, epoch * len(train_queue) + step
                    )
                    writer.add_scalar(
                        "LxBatch", objs_Lx.avg, epoch * len(train_queue) + step
                    )
                    writer.add_scalar(
                        "LzBatch", objs_Lz.avg, epoch * len(train_queue) + step
                    )
        if args.save:
            writer.add_scalar("LossEpoch", objs.avg, epoch)
            writer.add_scalar("LxEpoch", objs_Lx.avg, epoch)
            writer.add_scalar("LzEpoch", objs_Lz.avg, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        train_obj = objs.avg
        logging.info(f"[train] loss {train_obj:.3f} Lx {objs_Lx.avg:.3f} Lz {objs_Lz.avg:.3f}")
        scheduler.step()

        # save checkpoint
        is_best = False
        if train_obj < best_obj:
            best_obj = train_obj
            is_best = True
        utils.save_checkpoint(
            {
                "epoch": epoch,
                "best_obj": best_obj,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            args.savedir,
        )

        # generate images
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                batch_size = 64
                # only banned, size matter for captions
                caption_data = Captions(datadir=args.datadir, banned=banned, size=batch_size)
                caption_queue = DataLoader(dataset=caption_data, batch_size=batch_size, shuffle=False, drop_last=False)
                for step, data in enumerate(caption_queue):
                    data = data.to(device)
                    x = model.generate(data, batch_size)
                    
                    fig = plt.figure(figsize=(16, 16))
                    plt.axis("off")
                    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in x]
                    anim = animation.ArtistAnimation(
                        fig, ims, interval=500, repeat_delay=1000, blit=True
                    )
                    anim.save(
                        Path(args.savedir, f"draw_epoch_{epoch:d}.gif"),
                        dpi=100,
                        writer="imagemagick",
                    )
                    plt.close("all")


if __name__ == "__main__":
    main()
