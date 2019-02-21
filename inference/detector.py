import tensorflow as tf
import numpy as np


class Detector:
    def __init__(self, model_path, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/images:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/labels:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes_per_image:0')
        ]

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

    def __call__(self, image, score_threshold=0.1):
        """
        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            labels: an int numpy array of shape [num_faces].
            scores: a float numpy array of shape [num_faces].

        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """

        feed_dict = {self.input_image: np.expand_dims(image, 0)}
        boxes, labels, scores, n = self.sess.run(self.output_ops, feed_dict)

        n = n[0]
        to_keep = scores[:n] > score_threshold
        boxes = boxes[:n][to_keep]
        labels = labels[:n][to_keep]
        scores = scores[:n][to_keep]

        return boxes, labels, scores
