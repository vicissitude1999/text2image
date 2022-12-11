import os
import shutil

import numpy as np
import torch


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

        self.records = np.asarray([])
        self.std = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def update_std(self, val):
        self.records.append(val)
        self.std = np.std(self.records)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    if scripts_to_save:
        os.makedirs(os.path.join(path, "scripts"), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, is_best, save):
    os.makedirs(save, exist_ok=True)
    filename = os.path.join(save, "current.ckpt")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "best.ckpt")
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


#### helper functions from https://github.com/czm0/draw_pytorch/utils.py
#### not used currently
def unit_prefix(x, n=1):
    for i in range(n):
        x = x.unsqueeze(0)
    return x


def align(x, y, start_dim=0):
    xd, yd = x.dim(), y.dim()
    if xd > yd:
        y = unit_prefix(y, xd - yd)
    elif yd > xd:
        x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd - i - 1
        if ys[td] == 1:
            ys[td] = xs[td]
        elif xs[td] == 1:
            xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)


def matmul(X, Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i], Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)
