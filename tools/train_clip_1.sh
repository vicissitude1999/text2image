export CUDA_VISIBLE_DEVICES=6
# python src/train.py mnist tools/mnist-captions-clip.json clip AlignDrawClip
python src/train.py coco tools/coco-captions-clip.json clip AlignDrawClip
# python src/train.py coco tools/coco-captions-32x32.json
