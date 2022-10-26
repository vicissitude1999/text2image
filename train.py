import torch
import numpy as np
from random import randint
from torch.utils.data import DataLoader

import dataset
from dataset import MNISTCaptions

EPOCHS = 1000
BATCH_SIZE = 10


def main():
    banned = [randint(0, 10) for i in range(12)]
    dset = MNISTCaptions(banned=banned, train=True)
    dloader = DataLoader(dataset=dset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    for epoch in range(EPOCHS):
        for step, data in enumerate(dloader):
            image, caption = data
            print(caption.shape)
            return
    
if __name__ == "__main__":
    main()