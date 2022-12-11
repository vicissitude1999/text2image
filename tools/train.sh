export CUDA_VISIBLE_DEVICES=1
python src/train.py mnist tools/mnist-captions.json base AlignDraw
# python src/train.py coco tools/coco-captions-32x32.json base AlignDraw
# python src/train.py coco tools/coco-captions-32x32.json
