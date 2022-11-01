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

from dataset_clip import COCOCaptions, CaptionSameLenBatchSampler, MNISTCaptions, Captions
from model_clip import AlignDrawClip
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
    config = sys.argv[2]  # "tools/xx.json

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
    val_batchsize = 64 # number of samples used to generate images
    if sys.argv[1] == "coco":
        train_data = COCOCaptions(datadir=args.datadir, split="train")
        sampler = CaptionSameLenBatchSampler(train_data, args.batch_size, seed=args.seed)
        train_queue = DataLoader(dataset=train_data, batch_sampler=sampler)
        
        val_data = COCOCaptions(datadir=args.datadir, split="dev")
        val_sampler = CaptionSameLenBatchSampler(val_data, batch_size=val_batchsize, seed=args.seed)
        val_queue = DataLoader(dataset=val_data, batch_sampler=val_sampler)
    elif sys.argv[1] == "mnist":
        banned = [randint(0, 10) for _ in range(12)]
        train_data = MNISTCaptions(datadir=args.datadir, banned=banned, size=10000, train=True, seed=args.seed)
        train_queue = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        val_data = MNISTCaptions(datadir=args.datadir, banned=[], size=val_batchsize, train=False, seed=args.seed+1)
        val_queue = DataLoader(dataset=val_data, batch_size=val_batchsize, shuffle=False, drop_last=False)

    # model = AlignDraw
    model = AlignDrawClip(
        args.model[0].runSteps,
        args.model[0].dimReadAttent,
        args.model[0].dimWriteAttent,
        args.model[0].dimA,
        args.model[0].dimB,
        args.model[0].channels,
        args.model[0].dimY,
        args.model[0].dimLangRNN,
        args.model[0].dimRNNEnc,
        args.model[0].dimZ,
        args.model[0].dimRNNDec,
        args.model[0].dimAlign,
        device=device,
    ).to(device)
    model.apply(initialize_weights)

    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.reduceLRAfter], gamma=0.1)

    best_obj = float("inf")
    for epoch in range(args.epochs):
        logging.info(f"[epoch] {epoch:d}/{args.epochs}")

        # train
        objs = utils.AvgrageMeter()
        objs_Lx = utils.AvgrageMeter()
        objs_Lz = utils.AvgrageMeter()
        model.train()
        
        for step, data in enumerate(train_queue):
            t1 = time.time()
            image, caption = data
            
            image = image.to(device, non_blocking=True)
            caption = caption.to(device, non_blocking=True)

            optimizer.zero_grad()
            Lx, Lz = model.loss((image, caption))
            loss = Lx + Lz
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)  # clip norm before step!
            optimizer.step()

            n = caption.size(0)
            objs.update(loss.item(), n)
            objs_Lx.update(Lx.item(), n)
            objs_Lz.update(Lz.item(), n)

            if step % args.report_freq == 0:
                logging.info(f"train {step:03d}/{len(train_queue):03d} loss {objs.avg:.3f} Lx {objs_Lx.avg:.3f} Lz {objs_Lz.avg:.3f}")
                if args.save:
                    writer.add_scalar("LossBatch", objs.avg, epoch * len(train_queue) + step)
                    writer.add_scalar("LxBatch", objs_Lx.avg, epoch * len(train_queue) + step)
                    writer.add_scalar("LzBatch", objs_Lz.avg, epoch * len(train_queue) + step)
                
                t2 = time.time()
                if step // args.report_freq in [1,3,5]:
                    hrs = (t2-t1)/3600 * len(train_queue)
                    print(f"\t\testimated {hrs} hrs per epoch, {hrs*args.epochs} hrs all epochs")
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
        if (epoch + 1) % (args.epochs // 5) == 0 or (epoch + 1) == args.epochs:
            model.eval()
            if sys.argv[1] == "mnist":
                val_data.reset() # only need to reset seed for mnist
            with torch.no_grad():
                for step, data in enumerate(val_queue):
                    if step == 0:
                        image, caption = data
                        caption = caption.to(device)
                        x = model.generate(caption)
                        vutils.save_image(x[-1], Path(args.savedir, f"epoch_{epoch:d}.jpg"))

                        fig = plt.figure(figsize=(16, 16))
                        plt.axis("off")
                        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in x]
                        anim = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
                        anim.save(
                            Path(args.savedir, f"epoch_{epoch:d}.gif"),
                            dpi=100,
                            writer="imagemagick",
                        )
                        plt.close("all")
                    else:
                        break


if __name__ == "__main__":
    main()
