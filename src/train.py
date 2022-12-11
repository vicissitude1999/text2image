import argparse
import glob
import time
import json
import logging
import os
import sys
import shutil
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from addict import Dict
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from model_builder import BUILDER
import utils
from dataset import COCOCaptions, CaptionSameLenBatchSampler, MNISTCaptions


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
    
    dataset = sys.argv[1]
    model_type = sys.argv[3]  # clip/base
    args.savedir = os.path.join(args.savedir, sys.argv[4])
    # output directory
    if args.save:
        args.savedir = "{}/{}".format(args.savedir, time.strftime("%Y%m%d-%H%M%S"))
        utils.create_exp_dir(args.savedir, scripts_to_save=glob.glob("src/*.py"))
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
    if dataset == "coco":
        train_data = COCOCaptions(datadir=args.datadir, split="train", mode=model_type)
        sampler = CaptionSameLenBatchSampler(train_data, args.batch_size, seed=args.seed)
        train_queue = DataLoader(dataset=train_data, batch_sampler=sampler)
        
        val_data = COCOCaptions(datadir=args.datadir, split="dev", mode=model_type)
        val_sampler = CaptionSameLenBatchSampler(val_data, batch_size=val_batchsize, seed=args.seed)
        val_queue = DataLoader(dataset=val_data, batch_sampler=val_sampler)
    elif dataset == "mnist":
        banned = np.random.choice(10+1, size=12, replace=True)
        
        train_data = MNISTCaptions(datadir=args.datadir, banned=banned, size=10000, train=True, seed=args.seed, mode=model_type)
        train_queue = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        val_data = MNISTCaptions(datadir=args.datadir, banned=[], size=val_batchsize, train=False, seed=args.seed+1, mode=model_type)
        val_queue = DataLoader(dataset=val_data, batch_size=val_batchsize, shuffle=False, drop_last=False)
    # model name
    aligndraw = BUILDER[sys.argv[4]]
    model = aligndraw(
            args.model[0],
            device=device,
        ).to(device)
    model.apply(initialize_weights)
    
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.reduceLRAfter], gamma=0.1)

    best_obj = float("inf")
    for epoch in tqdm(range(args.epochs)):
        logging.info(f"[epoch] {epoch:d}/{args.epochs}")
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
            Lx, Lz = model.loss((image, caption), myloss=False)
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
                    writer.add_scalar("LossBatch/L", objs.avg, epoch * len(train_queue) + step)
                    writer.add_scalar("LossBatch/Lx", objs_Lx.avg, epoch * len(train_queue) + step)
                    writer.add_scalar("LossBatch/Lz", objs_Lz.avg, epoch * len(train_queue) + step)
        if args.save:
            writer.add_scalar("LossEpoch/L", objs.avg, epoch)
            writer.add_scalar("LossEpoch/Lx", objs_Lx.avg, epoch)
            writer.add_scalar("LossEpoch/Lz", objs_Lz.avg, epoch)
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
        if epoch == 0 or (epoch + 1) % (args.epochs // 5) == 0 or (epoch + 1) == args.epochs:
            model.eval()
            if dataset == "coco":
                val_sampler.reset()
            elif dataset == "mnist":
                val_data.reset()
            objs_psnr = utils.AvgrageMeter()
            with torch.no_grad():
                for step, (image, caption) in enumerate(val_queue):
                    # save groundtruth images and captions of the first batch
                    if step == 0 and epoch == 0:
                        grid = vutils.make_grid(image.view(-1, args.model[0].channels, args.model[0].dimB, args.model[0].dimA),
                                nrow=int(math.sqrt(val_batchsize)), pad_value=1)
                        vutils.save_image(grid, Path(args.savedir, "gt_imgs.jpg"))
                        
                        if model_type == 'clip':
                            continue
                        with open(Path(args.savedir, "gt_captions.txt"), "w") as f:
                            for cap in caption:
                                f.write(val_data.decodeCaption(cap.tolist()) + "\n")
                    # save generated images of the first batch
                    if step == 0:
                        caption = caption.to(device)
                        imgs = model.generate(caption)
                        grids = []
                        
                        for img in imgs:
                            grids.append(vutils.make_grid(img, nrow=int(math.sqrt(args.batch_size)), pad_value=1))
                        
                        vutils.save_image(grids[-1], Path(args.savedir, f"epoch_{epoch:d}.jpg"))
                        
                        fig = plt.figure(figsize=(16, 16))
                        plt.axis("off")
                        ims = [[plt.imshow(np.transpose(grid, (1, 2, 0)), animated=True)] for grid in grids]
                        anim = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
                        anim.save(
                            Path(args.savedir, f"epoch_{epoch:d}.gif"),
                            dpi=100,
                            writer="imagemagick",
                        )
                        plt.close("all")
                    # compute psnr
                    from skimage.metrics import peak_signal_noise_ratio
                    image = image.reshape(-1, 3, 32, 32).detach().cpu().numpy()
                    x_generated = imgs[-1].detach().cpu().numpy()
                    
                    psnr_list = np.zeros(image.shape[0])
                    for i in range(image.shape[0]):
                        true_img = image[i]
                        test_img = x_generated[i]
                        psnr = peak_signal_noise_ratio(true_img, test_img)
                        psnr_list[i] = psnr
                    test_psnr = psnr_list.mean()
                    objs_psnr.update(test_psnr, image.shape[0])
                    if args.save:
                        writer.add_scalar("PSNR", objs_psnr.avg, epoch * len(val_queue) + step)


if __name__ == "__main__":
    main()
