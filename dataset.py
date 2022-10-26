import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from random import randint
import numpy as np


class MNISTCaptions(Dataset):
    def __init__(self, banned, train=True) -> None:
        self.banned = banned
                
        self.dictionary = {"0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "the": 10, "digit": 11, "is": 12, "on": 13, "at": 14, "left": 15, "right": 16, "bottom": 17, "top": 18, "of": 19, "image": 20, ".": 21}
        self.reverse_dictionary = create_reverse_dictionary(self.dictionary)
        
        mnist = datasets.MNIST("../../data", train=train, download=True) # [PIL, tensor]
        self.data = mnist.data.numpy() # 28 * 28
        self.targets = mnist.targets.numpy()
        self.len = len(mnist)
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        while True:    
            k = randint(0, 7)
            i = randint(0, self.len)
            j = randint(0, self.len)
            
            # some cases are hidden from training set
            if k <= 3:
                if self.targets[i] == self.banned[k*2] or self.targets[j] == self.banned[k*2+1]:
                    continue
            else:
                if self.targets[i] == self.banned[k+4]:
                    continue
            
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
            
            if k == 0 or k == 1:
                image = create_2digit_mnist_image_leftright(self.data[i], self.data[j])
            if k == 2 or k == 3:
                image = create_2digit_mnist_image_topbottom(self.data[i], self.data[j])
            if k == 4:
                image = create_1digit_mnist_image_topleft(self.data[i])
            if k == 5:
                image = create_1digit_mnist_image_bottomright(self.data[i])
            if k == 6:
                image = create_1digit_mnist_image_topright(self.data[i])
            if k == 7:
                image = create_1digit_mnist_image_bottomleft(self.data[i])
            
            return image, caption


def create_reverse_dictionary(dictionary):
    dictionary_reverse = {}

    for word in dictionary:
        index = dictionary[word]
        dictionary_reverse[index] = word
    return dictionary_reverse

def sent2matrix(sentence, dictionary):
    words = sentence.split()
    m = np.int32(np.zeros(len(words))) 

    for i in range(len(words)):
        m[i] = dictionary[words[i]]

    return m

def matrix2sent(matrix, reverse_dictionary):
    text = ""
    for i in range(matrix.shape[0]):
        text = text + " " + reverse_dictionary[matrix[i]]

    return text

def create_2digit_mnist_image_leftright(digit1, digit2):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))

    w = randint(16,18)
    h = randint(0,4)
    image[w:w+28,h:h+28] = digit1

    h = randint(28,32)
    image[w:w+28,h:h+28] = digit2

    image = image.reshape(-1)

    return image

def create_2digit_mnist_image_topbottom(digit1, digit2):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))

    h = randint(16,18)
    w = randint(0,2)
    image[w:w+28,h:h+28] = digit1

    w = randint(30,32)
    image[w:w+28,h:h+28] = digit2

    image = image.reshape(-1)

    return image

def create_1digit_mnist_image_topleft(digit1):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))

    w = randint(0,2)
    h = randint(0,4)
    image[w:w+28,h:h+28] = digit1

    image = image.reshape(-1)

    return image

def create_1digit_mnist_image_topright(digit1):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))

    w = randint(0,2)
    h = randint(28,32)
    image[w:w+28,h:h+28] = digit1

    image = image.reshape(-1)

    return image

def create_1digit_mnist_image_bottomright(digit1):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))

    w = randint(30,32)
    h = randint(28,32)
    image[w:w+28,h:h+28] = digit1

    image = image.reshape(-1)

    return image

def create_1digit_mnist_image_bottomleft(digit1):
    """ Digits is list of numpy arrays, where each array is a digit"""
    
    image = np.zeros((60,60))

    w = randint(30,32)
    h = randint(0,4)
    image[w:w+28,h:h+28] = digit1

    image = image.reshape(-1)

    return image
