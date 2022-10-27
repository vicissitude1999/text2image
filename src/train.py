import argparse
import glob
import time
import json
import logging
import os
import sys
import random
from random import randint
import shutil

import numpy as np
from pathlib import Path
from addict import Dict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MNISTCaptions
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
        utils.create_exp_dir(args.savedir, scripts_to_save=None)
        with open(Path(args.savedir, "args.json"), "w") as f:
            json.dump(args, f)
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
        
    banned = [randint(0, 10) for i in range(12)]
    train_data = MNISTCaptions(datadir=args.datadir, banned=banned, train=True)
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
        device
    ).to(device)
    model.apply(initialize_weights)
    

    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.reduceLRAfter], gamma=0.1
    )

    best_obj = float("inf")
    for epoch in range(args.epochs):
        logging.info(f"epoch {epoch:d}")
        
        # train
        objs = utils.AvgrageMeter()
        model.train()
        
        for step, data in enumerate(train_queue):
            image, caption = data
            image = image.to(device, non_blocking=True)
            caption = caption.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = model.loss((image, caption))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            n = caption.size(0)
            objs.update(loss.item(), n)

            if step % args.report_freq == 0:
                logging.info(f"train {step:03d} loss {objs.avg:f}")
                if args.save:
                    writer.add_scalar(
                        "LossBatch/train", objs.avg, epoch * len(train_queue) + step
                    )
        if args.save:
            writer.add_scalar("LossEpoch/train", objs.avg, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            
        train_obj = objs.avg    
        logging.info(f"[train] loss {train_obj:f}")
        scheduler.step()

    is_best = False
    if train_obj < best_obj:
        best_obj = train_obj
        is_best = True

    # save checkpoint
    utils.save_checkpoint(
        {
            "epoch": epoch,
            "best_obj": best_obj,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        },
        is_best,
        args.save,
    )


if __name__ == "__main__":
    main()
