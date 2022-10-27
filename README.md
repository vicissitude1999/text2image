## text2image
PyTorch implementation of paper paper [Generating Images from Captions with Attention](http://arxiv.org/abs/1511.02793) by Elman Mansimov, Emilio Parisotto, Jimmy Ba and Ruslan Salakhutdinov; ICLR 2016.

### Getting Started

Download data:
For training mnist, PyTorch will download the raw data automatically.

Additionally, depending on the tasks you will probably need to download these files by running:

```
wget http://www.cs.toronto.edu/~emansim/datasets/mnist.h5
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-images-32x32.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-images-56x56.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-captions.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-captions-len.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/train-cap2im.pkl
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-images-32x32.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-images-56x56.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-captions.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-captions-len.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dev-cap2im.pkl
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/test-images-32x32.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/test-captions.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/test-captions-len.npy
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/test-cap2im.pkl
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/gan.hdf5
wget http://www.cs.toronto.edu/~emansim/datasets/text2image/dictionary.pkl
```

### MNIST with Captions

To train the model go to text2image folder and run

```
bash tools/train.sh
```

To generate 60x60 MNIST images from captions as specified in appendix of the paper run

### Microsoft COCO

### Reference

https://github.com/mansimov/text2image

https://github.com/Natsu6767/Generating-Devanagari-Using-DRAW/blob/master/draw_model.py
