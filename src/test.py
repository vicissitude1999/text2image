import argparse
import time
import json
import math
import random
import h5py

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
from dataset import COCOCaptions, CaptionSameLenBatchSampler, MNISTCaptions
from model_builder import BUILDER
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="validation or test", default="validation")
    parser.add_argument("--train_dir", type=str, help="directory of output from training")
    parser.add_argument("--caption_path", type=str, help="path to caption file", default="")
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


def load_weights(params, path, num_conv):
    with h5py.File(path, "r") as hdf5:
        params["skipthought2image"] = np.copy(hdf5["skipthought2image"])
        params["skipthought2image-bias"] = np.copy(hdf5["skipthought2image-bias"])

        for i in range(num_conv):
            params[f"W_conv{i}"] = np.copy(hdf5["W_conv{}".format(i)])
            params[f"b_conv{i}"] = np.copy(hdf5["b_conv{}".format(i)])

            # Flip w, h axes
            params[f"W_conv{i}"] = params[f"W_conv{i}"][:,:,::-1,::-1]

            w = np.abs(np.copy(hdf5[f"W_conv{i}"]))
            print(f"W_conv{i}", np.min(w), np.mean(w), np.max(w))
            b = np.abs(np.copy(hdf5[f"b_conv{i}"]))
            print(f"b_conv{i}", np.min(b), np.mean(b), np.max(b))

    return params


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
    
    
    # model name
    aligndraw = BUILDER[args.name]
    model = aligndraw(
            args.model[0],
            device=device,
        ).to(device)
    ckpt = torch.load(Path(args.train_dir, "current.ckpt"))
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    if args.mode == "test":
        # create dataset
        if args.dataset == "coco":
            val_data = COCOCaptionsOnly(caption_path=args.caption_path, datadir=args.datadir, mode=args.model_type)
            val_sampler = CaptionSameLenBatchSampler(val_data, batch_size=args.batch_size, seed=args.seed)
            # Input caption file needs to have equal length captions
            val_queue = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
        elif args.dataset == "mnist":
            val_data = MNISTCaptionsOnly(caption_path=args.caption_path, mode=args.model_type)
            val_queue = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
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
    elif args.mode == "validation":
        if args.dataset == "coco":
            val_data = COCOCaptions(datadir=args.datadir, split="dev", mode=args.model_type)
            val_sampler = CaptionSameLenBatchSampler(val_data, batch_size=args.batch_size, seed=args.seed)
            val_queue = DataLoader(dataset=val_data, batch_sampler=val_sampler)
        elif args.dataset == "mnist":
            val_data = MNISTCaptions(datadir=args.datadir, banned=[], size=args.batch_size, train=False, seed=args.seed+1, mode=args.model_type)
            val_queue = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        objs_psnr = utils.AvgrageMeter()

        with torch.no_grad():
            for step, (image, caption) in enumerate(val_queue):
                caption = caption.to(device)
                imgs = model.generate(caption)
                
                # compute psnr
                from skimage.metrics import peak_signal_noise_ratio
                if args.dataset == "mnist":
                    image = image.detach().reshape(-1, 1, 60, 60)
                elif args.dataset == "coco":
                    image = image.detach().reshape(-1, 3, 32, 32)
                else:
                    raise NameError("Invalid dataset name")
                x_generated = imgs[-1].detach()
                
                psnr_list = np.zeros(image.shape[0])
                for i in range(image.shape[0]):
                    true_img = image[i]
                    test_img = x_generated[i]
                    psnr = peak_signal_noise_ratio(true_img.cpu().numpy(), test_img.cpu().numpy())
                    psnr_list[i] = psnr
                test_psnr = psnr_list.mean()
                objs_psnr.update(test_psnr, image.shape[0])
                
                # some visualizations
                # save groundtruth images and captions of the first batch
                grid = vutils.make_grid(image, nrow=int(math.sqrt(args.batch_size)), pad_value=1)
                vutils.save_image(grid, Path(args.savedir, f"gt_imgs_{step}.jpg"))
                
                if args.model_type == 'clip':
                    continue
                with open(Path(args.savedir, f"gt_captions_{step}.txt"), "w") as f:
                    for cap in caption:
                        f.write(val_data.decodeCaption(cap.tolist()) + "\n")
                        
                # save generated images
                # print(imgs[-1].shape)
                # print(int(math.sqrt(args.batch_size)))
                
                grid = vutils.make_grid(imgs[-1], nrow=int(math.sqrt(args.batch_size)), pad_value=1)
                vutils.save_image(grid, Path(args.savedir, f"generated_imgs_{step}.jpg"))
            with open(Path(args.savedir, f"psnr.txt"), "w") as f:
                f.write(str(objs_psnr.avg))


if __name__ == "__main__":
    main()
