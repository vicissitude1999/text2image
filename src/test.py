import argparse
import time
import json
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from addict import Dict

import torch
import torch.backends.cudnn as cudnn
import torch.utils
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from dataset import COCOCaptionsOnly, CaptionSameLenBatchSampler, MNISTCaptionsOnly
from model_builder import BUILDER
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="directory of output from training")
    parser.add_argument("--caption_path", type=str, help="path to caption file")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--model_type", type=str, default="base")
    parser.add_argument("--name", type=str, default="AlignDraw")
    
    opt = parser.parse_args()
    
    return opt

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

def main():
    opt = parse_args()
    # load training params
    with open(Path(opt.train_dir, "args.json")) as f:
        args = Dict(json.load(f))
    # merge training and test params
    for key, value in vars(opt).items():
        args[key] = value
    print(args)
    
    # initialize seed
    init_seeds(args.seed, False)
    device = "cuda"
    
    # output directory 
    args.savedir = "{}/test-{}".format(args.train_dir, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.savedir, scripts_to_save=None)
    
    # create dataset
    if args.dataset == "coco":
        val_data = COCOCaptionsOnly(caption_path=args.caption_path, datadir=args.datadir, mode=args.model_type)
        val_sampler = CaptionSameLenBatchSampler(val_data, batch_size=args.batch_size, seed=args.seed)
        # need to save the original order
        # TODO make it work for input captions with difference lengths
        # batch_indices = []
        # for ind in val_sampler:
        #     batch_indices.append(ind)
        # val_queue = DataLoader(dataset=val_data, batch_sampler=val_sampler)
        val_queue = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    elif args.dataset == "mnist":
        val_data = MNISTCaptionsOnly(caption_path=args.caption_path, mode=args.model_type)
        val_queue = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
    # model name
    aligndraw = BUILDER[args.name]
    model = aligndraw(
            args.model[0],
            device=device,
        ).to(device)
    ckpt = torch.load(Path(args.train_dir, "current.ckpt"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    with torch.no_grad():
        for step, caption in enumerate(val_queue):
            caption = caption.to(device)
            imgs = model.generate(caption)
            grids = []
            
            for img in imgs:
                # if args.dataset == "coco":
                #     img = img[batch_indices[step]]
                grids.append(vutils.make_grid(img, nrow=int(math.sqrt(args.batch_size)), pad_value=1))
        
            vutils.save_image(grids[-1], Path(args.savedir, f"batch_{step:d}.jpg"))
            
            fig = plt.figure(figsize=(16, 16))
            plt.axis("off")
            ims = [[plt.imshow(np.transpose(grid, (1, 2, 0)), animated=True)] for grid in grids]
            anim = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
            anim.save(
                Path(args.savedir, f"batch_{step:d}.gif"),
                dpi=100,
                writer="imagemagick",
            )
            plt.close("all")


if __name__ == "__main__":
    main()
