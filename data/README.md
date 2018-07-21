
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