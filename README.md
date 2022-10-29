## text2image
PyTorch implementation of paper paper [Generating Images from Captions with Attention](http://arxiv.org/abs/1511.02793) by Elman Mansimov, Emilio Parisotto, Jimmy Ba and Ruslan Salakhutdinov; ICLR 2016.

### Getting Started

Download data:
For training mnist, PyTorch will download the raw data automatically.

Additionally, depending on the tasks you will probably need to download these files by running:

Install COCO API:
```
conda install cython
git clone git@github.com:cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make
python setup.py install
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
