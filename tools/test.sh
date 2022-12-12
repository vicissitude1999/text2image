python src/test.py --train_dir outputs/mnist/AlignDraw/20221211-220930 \
--caption_path tools/captions_mnist.txt \
--dataset mnist \
--batch_size 100 \
--model_type base

python src/test.py --train_dir outputs/coco32/20221101-200132 \
--caption_path tools/captions_coco.txt \
--dataset coco \
--batch_size 25 \
--model_type base



python src/test.py --train_dir outputs/mnist_clip/20221101-190647 \
--caption_path tools/captions_mnist.txt \
--dataset mnist \
--batch_size 25 \
--model_type clip

python src/test.py --train_dir outputs/coco32_clip/20221101-203543 \
--caption_path tools/captions_coco.txt \
--dataset coco \
--batch_size 25 \
--model_type clip