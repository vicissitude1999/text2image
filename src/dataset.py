from collections import defaultdict
import math

import torch
from torch.utils.data import Dataset, Sampler
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from random import randint
import numpy as np
import pickle
from pathlib import Path


class COCOCaptions(Dataset):
    # split: train, dev, test
    def __init__(self, datadir, split) -> None:
        super().__init__()

        # image data, n/5 * 3072 (=32*32*3)  each image has 5 captions
        data_path = Path(datadir, f"{split}-images-32x32.npy")
        # caption data, n * 57 (max length of caption)
        captions_path = Path(datadir, f"{split}-captions.npy")
        # length of captions # n * 1
        captions_len = Path(datadir, f"{split}-captions-len.npy")
        # mapping from caption index to image index # (essentially just i//5)
        cap2im = Path(datadir, f"{split}-cap2im.pkl")
        # dictionary
        dictionary = Path(datadir, "dictionary.pkl")

        self.data = np.load(data_path).astype(np.float32)
        self.captions = np.load(captions_path).astype(np.int64)
        self.captions_len = np.load(captions_len).astype(np.int32).reshape(-1)
        with open(cap2im, "rb") as f:
            self.cap2im = pickle.load(f)
        with open(dictionary, "rb") as f:
            self.dictionary = pickle.load(f)
        self.reverse_dictionary = create_reverse_dictionary(self.dictionary)

    def __len__(self):
        return self.captions.shape[0]

    def __getitem__(self, index):
        # by using the custom batch sampler defined below, all captions within a batch have the same length,
        # so we don't need to use collate_fn to pad shorter captions
        return self.data[self.cap2im[index]], self.captions[index][0 : self.captions_len[index]]

    def decodeCaption(self, caption):
        return " ".join([self.reverse_dictionary[c] for c in caption])


# custom batch sampler to make sure captions within a batch have the same length
# the logic is identical to homogeneous-data.py in mansimov's code
class CaptionSameLenBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=5, minlen=7, maxlen=30) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

        captions_len = dataset.captions_len
        self.len_to_indices = defaultdict(list)
        self.rng = np.random.default_rng(seed)

        for i, l in enumerate(captions_len):
            if (minlen is None or l >= minlen) and (maxlen is None or l <= maxlen):
                self.len_to_indices[l].append(i)
        # have to compute the length after filtering by minlen and maxlen
        self.size = 0
        for l in self.len_to_indices:
            self.size += math.ceil(len(self.len_to_indices[l]) / self.batch_size)

    def __len__(self):
        return self.size

    def __iter__(self):
        batches = []

        for l in self.len_to_indices:
            self.rng.shuffle(self.len_to_indices[l])
            # torch.split doesn't work the same as np.split
            for batch in torch.split(torch.tensor(self.len_to_indices[l]), self.batch_size):
                batches.append(batch.tolist())
        self.rng.shuffle(batches)

        return iter(batches)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)


class MNISTCaptions(Dataset):
    def __init__(self, datadir, banned, size=10000, train=True, seed=5) -> None:
        self.banned = banned
        self.train = train
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.dictionary = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "the": 10,
            "digit": 11,
            "is": 12,
            "on": 13,
            "at": 14,
            "left": 15,
            "right": 16,
            "bottom": 17,
            "top": 18,
            "of": 19,
            "image": 20,
            ".": 21,
        }
        self.reverse_dictionary = create_reverse_dictionary(self.dictionary)

        # may use transforms.ToTensor to convert PIL to 0-1 tensor
        mnist = datasets.MNIST(datadir, train=train, download=True)
        self.data = mnist.data.numpy() / 255  # 28 * 28, 0 - 255
        self.targets = mnist.targets.numpy()
        self.size = size
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        while True:
            k = self.rng.choice(7+1)
            i = self.rng.choice(self.size+1)
            j = self.rng.choice(self.size+1)

            image = self.getImage(k, i, j)
            caption = self.getCaption(k, i, j)
            if caption is None:
                continue

            return image, caption
        
    # reset the rng
    # for example, when generating images, we want to generate the same set of data at each epoch
    def reset(self):
        self.rng = np.random.default_rng(self.seed)
    
    
    def getImage(self, k, i, j):
        if k == 0 or k == 1:
            image = create_2digit_mnist_image_leftright(self.data[i], self.data[j], self.rng)
        if k == 2 or k == 3:
            image = create_2digit_mnist_image_topbottom(self.data[i], self.data[j], self.rng)
        if k == 4:
            image = create_1digit_mnist_image_topleft(self.data[i], self.rng)
        if k == 5:
            image = create_1digit_mnist_image_bottomright(self.data[i], self.rng)
        if k == 6:
            image = create_1digit_mnist_image_topright(self.data[i], self.rng)
        if k == 7:
            image = create_1digit_mnist_image_bottomleft(self.data[i], self.rng)

        return image

    def getCaption(self, k, i, j):
        # some cases are hidden from training set
        if self.train:
            if k <= 3:
                if self.targets[i] == self.banned[k * 2] or self.targets[j] == self.banned[k * 2 + 1]:
                    return None
            else:
                if self.targets[i] == self.banned[k + 4]:
                    return None

        if k == 0:
            sentence = "the digit %d is on the left of the digit %d ." % (self.targets[i], self.targets[j])
        elif k == 1:
            sentence = "the digit %d is on the right of the digit %d ." % (self.targets[j], self.targets[i])
        elif k == 2:
            sentence = "the digit %d is at the top of the digit %d ." % (self.targets[i], self.targets[j])
        elif k == 3:
            sentence = "the digit %d is at the bottom of the digit %d ." % (self.targets[j], self.targets[i])
        elif k == 4:
            sentence = "the digit %d is at the top left of the image ." % (self.targets[i])
        elif k == 5:
            sentence = "the digit %d is at the bottom right of the image ." % (self.targets[i])
        elif k == 6:
            sentence = "the digit %d is at the top right of the image ." % (self.targets[i])
        elif k == 7:
            sentence = "the digit %d is at the bottom left of the image ." % (self.targets[i])

        caption = sent2matrix(sentence, self.dictionary)

        return caption
    
    def decodeCaption(self, caption):
        return " ".join([self.reverse_dictionary[c] for c in caption])


def create_reverse_dictionary(dictionary):
    dictionary_reverse = {}

    for word in dictionary:
        index = dictionary[word]
        dictionary_reverse[index] = word
    return dictionary_reverse


def sent2matrix(sentence, dictionary):
    words = sentence.split()
    m = np.zeros(len(words), dtype=np.int64)  # int64 required for using torch.nn.functional.on_hot

    for i in range(len(words)):
        m[i] = dictionary[words[i]]

    return m


def matrix2sent(matrix, reverse_dictionary):
    text = ""
    for i in range(matrix.shape[0]):
        text = text + " " + reverse_dictionary[matrix[i]]

    return text


def create_2digit_mnist_image_leftright(digit1, digit2, rng):
    """Digits is list of numpy arrays, where each array is a digit"""

    image = np.zeros((60, 60), dtype=np.float32)

    w = rng.choice(range(16, 18+1))
    h = rng.choice(range(0, 4+1))
    image[w : w + 28, h : h + 28] = digit1

    h = randint(28, 32)
    image[w : w + 28, h : h + 28] = digit2

    image = image.reshape(-1)

    return image


def create_2digit_mnist_image_topbottom(digit1, digit2, rng):
    """Digits is list of numpy arrays, where each array is a digit"""

    image = np.zeros((60, 60), dtype=np.float32)

    h = rng.choice(range(16, 18+1))
    w = rng.choice(range(0, 2+1))
    image[w : w + 28, h : h + 28] = digit1

    w = randint(30, 32)
    image[w : w + 28, h : h + 28] = digit2

    image = image.reshape(-1)

    return image


def create_1digit_mnist_image_topleft(digit1, rng):
    """Digits is list of numpy arrays, where each array is a digit"""

    image = np.zeros((60, 60), dtype=np.float32)

    w = rng.choice(range(0, 2+1))
    h = rng.choice(range(0, 4+1))
    image[w : w + 28, h : h + 28] = digit1

    image = image.reshape(-1)

    return image


def create_1digit_mnist_image_topright(digit1, rng):
    """Digits is list of numpy arrays, where each array is a digit"""

    image = np.zeros((60, 60), dtype=np.float32)

    w = rng.choice(range(0, 2+1))
    h = rng.choice(range(28, 32+1))
    image[w : w + 28, h : h + 28] = digit1

    image = image.reshape(-1)

    return image


def create_1digit_mnist_image_bottomright(digit1, rng):
    """Digits is list of numpy arrays, where each array is a digit"""

    image = np.zeros((60, 60), dtype=np.float32)

    w = rng.choice(range(30, 32+1))
    h = rng.choice(range(28, 32+1))
    image[w : w + 28, h : h + 28] = digit1

    image = image.reshape(-1)

    return image


def create_1digit_mnist_image_bottomleft(digit1, rng):
    """Digits is list of numpy arrays, where each array is a digit"""

    image = np.zeros((60, 60), dtype=np.float32)

    w = rng.choice(range(30, 32+1))
    h = rng.choice(range(0, 4+1))
    image[w : w + 28, h : h + 28] = digit1

    image = image.reshape(-1)

    return image


if __name__ == "__main__":
    # cap = datasets.CocoCaptions(root = "data/coco/train2014/",
    #                     annFile = "data/coco/annotations/captions_train2014.json",
    #                     transform=transforms.PILToTensor())
    # img, target = cap[0]

    # print("Image Size: ", img.size())
    # print(target)

    def towords(s, reverse_dict):
        return " ".join([reverse_dict[e] for e in s if e > 0])

    data = np.float32(np.load("data/train-images-32x32.npy"))
    data_captions = np.int32(np.load("data/train-captions.npy"))
    data_captions_len = np.int32(np.load("data/train-captions-len.npy"))
    with open("data/train-cap2im.pkl", "rb") as f:
        data_cap2im = pickle.load(f)

    print(data.shape)  # n * 3072
    print(data_captions.shape)  # (5*n) * max(seq)
    print(data_captions_len.shape)  # (5*n) * 1
    for key in data_cap2im:
        assert data_cap2im[key] == key // 5
    # print(data_cap2im[5])

    with open("data/dictionary.pkl", "rb") as f:
        dictionary = pickle.load(f)
    print(len(list(dictionary.keys())))
    reverse_dictionary = create_reverse_dictionary(dictionary)

    for i in range(10):
        print(towords(data_captions[i], reverse_dictionary))
