## text2image
PyTorch implementation of paper paper [Generating Images from Captions with Attention](http://arxiv.org/abs/1511.02793) by Elman Mansimov, Emilio Parisotto, Jimmy Ba and Ruslan Salakhutdinov; ICLR 2016.

### Getting Started

### MNIST with Captions
To train MNIST, run
```
bash tools/train.sh # AlignDRAW
bash tools/train_clip.sh #clip_AlignDRAW
```
To test MNIST, run
```
python src/test.py --train_dir --caption_path --dataset --batch_size --model_type
```
Examples are in tools/test.sh

### COCO with Captions
Download the following to text2image/data/
```
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
To train COCO, run
```
bash tools/train.sh # AlignDRAW
bash tools/train_clip.sh #clip_AlignDRAW
```
To validate COCO, run
```
python src/test.py --train_dir --dataset --batch_size --model_type --name
```
To test COCO on any captions, provide a file of captions like tools/captions_mnist.txt
and run 
```
python src/test.py --mode test --train_dir --caption_path --dataset --batch_size --model_type --name
```
Make sure that all captions in the file have the same length.
Some examples are in tools/test.sh

### Reference

https://github.com/mansimov/text2image

https://github.com/Natsu6767/Generating-Devanagari-Using-DRAW/blob/master/draw_model.py
