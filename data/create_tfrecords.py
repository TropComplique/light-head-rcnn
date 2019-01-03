import io
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import json
import shutil
import random
import math
import argparse
from tqdm import tqdm


"""
The purpose of this script is to create a set of .tfrecords files
from a folder of images and a folder of annotations.
Annotations are in the json format.
Images must have .jpg or .jpeg filename extension.

Example of a json annotation (with filename "132416.json"):
{
  "object": [
    {"bndbox": {"ymin": 20, "ymax": 276, "xmax": 1219, "xmin": 1131}, "name": "dog"},
    {"bndbox": {"ymin": 1, "ymax": 248, "xmax": 1149, "xmin": 1014}, "name": "face"}
  ],
  "filename": "132416.jpg",
  "size": {"depth": 3, "width": 1920, "height": 1080}
}

`labels` text file contains a list of all label names.
One label name per line. Number of line - label encoding by an integer.

Example of use:
python create_tfrecords.py \
    --image_dir=/mnt/datasets/dan/wider_train/images/ \
    --annotations_dir=/mnt/datasets/dan/wider_train/annotations/ \
    --output=/mnt/datasets/dan/wider_train_shards/ \
    --labels=wider_labels.txt \
    --num_shards=100
"""


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_dir', type=str)
    parser.add_argument('-a', '--annotations_dir', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-l', '--labels', type=str)
    parser.add_argument('-s', '--num_shards', type=int, default=1)
    return parser.parse_args()


def dict_to_tf_example(annotation, image_dir, label_encoder):
    """Convert dict to tf.Example proto.

    Notice that this function normalizes the bounding
    box coordinates provided by the raw data.

    Arguments:
        annotation: a dict.
        image_dir: a string, path to the image directory.
        label_encoder: a dict, class name -> integer.
    Returns:
        an instance of tf.Example.
    """
    image_name = annotation['filename']
    assert image_name.endswith('.jpg') or image_name.endswith('.jpeg')

    image_path = os.path.join(image_dir, image_name)
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.mode == 'L':  # if grayscale
        rgb_image = np.stack(3*[np.array(image)], axis=2)
        encoded_jpg = to_jpeg_bytes(rgb_image)
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
    elif image.mode != 'RGB':
        return None
    if image.format != 'JPEG':
        return None
    assert image.mode == 'RGB'

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])
    assert width > 1 and height > 1
    assert image.size[0] == width and image.size[1] == height
    ymin, xmin, ymax, xmax, labels = [], [], [], [], []

    for obj in annotation['object']:

        # it is assumed that all box coordinates are in
        # ranges [0, height] and [0, width]

        a = float(obj['bndbox']['ymin'])/height
        b = float(obj['bndbox']['xmin'])/width
        c = float(obj['bndbox']['ymax'])/height
        d = float(obj['bndbox']['xmax'])/width
        label = label_encoder[obj['name']]

        assert (a < c) and (b < d)
        assert (a <= 1.0) and (a >= 0.0)
        assert (b <= 1.0) and (b >= 0.0)
        assert (c <= 1.0) and (c >= 0.0)
        assert (d <= 1.0) and (d >= 0.0)

        ymin.append(a)
        xmin.append(b)
        ymax.append(c)
        xmax.append(d)
        labels.append(label)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(encoded_jpg),
        'xmin': _float_list_feature(xmin),
        'xmax': _float_list_feature(xmax),
        'ymin': _float_list_feature(ymin),
        'ymax': _float_list_feature(ymax),
        'labels': _int64_list_feature(labels)
    }))
    return example


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def to_jpeg_bytes(array):
    image = Image.fromarray(array)
    tmp = io.BytesIO()
    image.save(tmp, format='jpeg')
    return tmp.getvalue()


def main():
    ARGS = make_args()

    with open(ARGS.labels, 'r') as f:
        labels = {line.strip(): i for i, line in enumerate(f.readlines()) if line.strip()}
    assert len(labels) > 0
    print('Possible labels (and label encoding):', labels)
    print('Number of classes:', len(labels), '\n')

    image_dir = ARGS.image_dir
    annotations_dir = ARGS.annotations_dir
    print('Reading images from:', image_dir)
    print('Reading annotations from:', annotations_dir, '\n')

    examples_list = os.listdir(annotations_dir)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    print('Number of images:', num_examples)

    num_shards = ARGS.num_shards
    shard_size = math.ceil(num_examples/num_shards)
    print('Number of images per shard:', shard_size)

    output_dir = ARGS.output
    shutil.rmtree(output_dir, ignore_errors=True)
    os.mkdir(output_dir)

    shard_id = 0
    num_examples_written = 0
    num_skipped_images = 0
    for example in tqdm(examples_list):

        if num_examples_written == 0:
            shard_path = os.path.join(output_dir, 'shard-%04d.tfrecords' % shard_id)
            writer = tf.python_io.TFRecordWriter(shard_path)

        path = os.path.join(annotations_dir, example)
        annotation = json.load(open(path))
        tf_example = dict_to_tf_example(annotation, image_dir, label_encoder=labels)
        if tf_example is None:
            num_skipped_images += 1
            continue
        writer.write(tf_example.SerializeToString())
        num_examples_written += 1

        if num_examples_written == shard_size:
            shard_id += 1
            num_examples_written = 0
            writer.close()

    # this happens if num_examples % num_shards != 0
    if num_examples_written != 0:
        writer.close()

    print('Result is here:', ARGS.output)
    print('Number of skipped images:', num_skipped_images)


main()
