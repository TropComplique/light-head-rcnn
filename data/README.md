# Data preparation
Here are examples of commands for preparing training/evaluation data. To them you will need to
### WIDER

1. Download the dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).

2. Run `prepare_WIDER.ipynb` to clean and convert the dataset.

3. Create tfrecords
  ```
  python create_tfrecords.py \
      --image_dir=/mnt/datasets/dan/wider_train/images/ \
      --annotations_dir=/mnt/datasets/dan/wider_train/annotations/ \
      --output=/mnt/datasets/dan/wider_train_shards/ \
      --labels=wider_labels.txt \
      --num_shards=100

  python create_tfrecords.py \
      --image_dir=/mnt/datasets/dan/wider_val/images/ \
      --annotations_dir=/mnt/datasets/dan/wider_val/annotations/ \
      --output=/mnt/datasets/dan/wider_val_shards/ \
      --labels=wider_labels.txt \
      --num_shards=1
  ```

### COCO

1. Download train and val images [here](http://cocodataset.org/#download)
2. Get [COCO API](https://github.com/cocodataset/cocoapi)

3. Connvert tfrecords  
  ```
  python create_tfrecords.py \
      --image_dir=/mnt/datasets/dan/coco_train/images/ \
      --annotations_dir=/mnt/datasets/dan/coco_train/annotations/ \
      --output=/mnt/datasets/dan/coco_train_shards/ \
      --labels=coco_labels.txt \
      --num_shards=1000

  python create_tfrecords.py \
      --image_dir=/mnt/datasets/dan/coco_val/images/ \
      --annotations_dir=/mnt/datasets/dan/coco_val/annotations/ \
      --output=/mnt/datasets/dan/coco_val_shards/ \
      --labels=coco_labels.txt \
      --num_shards=1
  ```  
