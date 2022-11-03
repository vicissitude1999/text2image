# python src/test.py --train_dir outputs/mnist/20221101-190524 \
# --caption_path tools/captions_mnist.txt \
# --dataset mnist \
# --batch_size 100 \
# --model_type base

python src/test.py --train_dir outputs/coco32/20221101-200132 \
--caption_path tools/captions_coco.txt \
--dataset coco \
--batch_size 100 \
--model_type base