import numpy as np
import tensorflow as tf


"""
For evaluation during the training I use average precision @ iou=0.5
like in PASCAL VOC Challenge (detection task):
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf

But after the training I test trained models
using the official evaluation scripts.
"""


class Evaluator:
    """It creates ops like in tf.metrics API."""

    def __init__(self, num_classes):
        """
        Arguments:
            num_classes: an integer.
        """
        assert num_classes > 0
        self.num_classes = num_classes
        self.initialize()

    def evaluate(self, iou_threshold=0.5):

        self.metrics = {}
        for label in range(self.num_classes):
            self.metrics[label] = evaluate_detector(
                self.groundtruth[label],
                self.detections[label],
                iou_threshold
            )

        if self.num_classes > 1:
            APs = [
                self.metrics[label]['AP']
                for label in range(self.num_classes)
            ]
            self.metrics['mAP'] = np.mean(APs)

    def get_metric_ops(self, groundtruth, predictions):
        """
        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [1, N, 4].
                'labels': an int tensor with shape [1, N].
            predictions: a dict with the following keys
                'boxes': a float tensor with shape [M, 4].
                'labels': an int tensor with shape [M].
                'scores': a float tensor with shape [M].
        """

        def update_op_func(gt_boxes, gt_labels, boxes, labels, scores):
            image_name = '{}'.format(self.unique_image_id)
            self.unique_image_id += 1
            self.add_groundtruth(image_name, gt_boxes, gt_labels)
            self.add_detections(image_name, boxes, labels, scores)

        tensors = [
            groundtruth['boxes'][0], groundtruth['labels'][0],
            predictions['boxes'], predictions['labels'], predictions['scores']
        ]
        update_op = tf.py_func(update_op_func, tensors, [])

        def evaluate_func():
            self.evaluate()
            self.initialize()
        evaluate_op = tf.py_func(evaluate_func, [], [])

        def get_value_func(label, measure):
            def value_func():
                return np.float32(self.metrics[label][measure])
            return value_func

        with tf.control_dependencies([evaluate_op]):

            metric_names = [
                'AP', 'precision', 'recall', 'mean_iou_for_TP',
                'best_threshold', 'total_FP', 'total_FN'
            ]

            eval_metric_ops = {}

            if self.num_classes == 1:
                for measure in metric_names:
                    name = 'metrics/' + measure
                    value_op = tf.py_func(get_value_func(0, measure), [], tf.float32)
                    eval_metric_ops[name] = (value_op, update_op)

            if self.num_classes > 1:
                get_map = lambda: np.float32(self.metrics['mAP'])
                value_op = tf.py_func(get_map, [], tf.float32)
                eval_metric_ops['metrics/mAP'] = (value_op, update_op)

        return eval_metric_ops

    def initialize(self):
        # detections are separated by label
        self.detections = {label: [] for label in range(self.num_classes)}

        # groundtruth boxes are separated by label and by image
        self.groundtruth = {label: {} for label in range(self.num_classes)}

        # i will use this counter as an unique image identifier
        self.unique_image_id = 0

    def add_detections(self, image_name, boxes, labels, scores):
        """
        Arguments:
            image_name: a numpy string array with shape [].
            boxes: a numpy float array with shape [M, 4].
            labels: a numpy int array with shape [M].
            scores: a numpy float array with shape [M].
        """
        for box, label, score in zip(boxes, labels, scores):
            self.detections[label].append(get_box(box, image_name, score))

    def add_groundtruth(self, image_name, boxes, labels):
        for box, label in zip(boxes, labels):
            g = self.groundtruth[label]
            if image_name in g:
                g[image_name] += [get_box(box)]
            else:
                g[image_name] = [get_box(box)]


def get_box(box, image_name=None, score=None):
    ymin, xmin, ymax, xmax = box
    dictionary = {
        'ymin': ymin, 'xmin': xmin,
        'ymax': ymax, 'xmax': xmax,
    }

    # groundtruth and predicted boxes
    # have different format
    is_prediction = (score is not None)\
        and (image_name is not None)
    is_groundtruth = not is_prediction

    if is_prediction:
        dictionary['image_name'] = image_name
        dictionary['confidence'] = score
    elif is_groundtruth:
        dictionary['is_matched'] = False

    return dictionary


def evaluate_detector(groundtruth, detections, iou_threshold=0.5):
    """
    Arguments:
        groundtruth: a dict of lists with boxes,
            image -> list of groundtruth boxes on the image.
        detections: a list of boxes.
        iou_threshold: a float number.
    Returns:
        a dict with seven values.
    """

    # each ground truth box is either TP or FN
    num_groundtruth_boxes = 0

    for boxes in groundtruth.values():
        num_groundtruth_boxes += len(boxes)
    num_groundtruth_boxes = max(num_groundtruth_boxes, 1)

    # sort by confidence in decreasing order
    detections.sort(key=lambda box: box['confidence'], reverse=True)

    num_correct_detections = 0
    num_detections = 0
    mean_iou = 0.0
    precision = [0.0]*len(detections)
    recall = [0.0]*len(detections)
    confidences = [box['confidence'] for box in detections]

    for k, detection in enumerate(detections):

        # each detection is either TP or FP
        num_detections += 1

        groundtruth_boxes = groundtruth.get(detection['image_name'], [])
        best_groundtruth_i, max_iou = match(detection, groundtruth_boxes)

        if best_groundtruth_i >= 0 and max_iou >= iou_threshold:
            box = groundtruth_boxes[best_groundtruth_i]
            if not box['is_matched']:
                box['is_matched'] = True
                num_correct_detections += 1  # increase number of TP
                mean_iou += max_iou

        precision[k] = num_correct_detections/num_detections  # TP/(TP + FP)
        recall[k] = num_correct_detections/num_groundtruth_boxes  # TP/(TP + FN)

    ap = compute_ap(precision, recall)
    best_threshold, best_precision, best_recall = compute_best_threshold(
        precision, recall, confidences
    )
    mean_iou /= max(num_correct_detections, 1)

    return {
        'AP': ap, 'precision': best_precision,
        'recall': best_recall, 'best_threshold': best_threshold,
        'mean_iou_for_TP': mean_iou, 'total_FP': num_detections - num_correct_detections,
        'total_FN': num_groundtruth_boxes - num_correct_detections
    }


def compute_best_threshold(precision, recall, confidences):
    """
    Arguments:
        precision, recall, confidences: lists of floats of the same length.
    Returns:
        1. a float number, best confidence threshold.
        2. a float number, precision at the threshold.
        3. a float number, recall at the threshold.
    """
    if len(confidences) == 0:
        return 0.0, 0.0, 0.0

    precision = np.array(precision)
    recall = np.array(recall)
    confidences = np.array(confidences)

    diff = np.abs(precision - recall)
    prod = precision*recall
    best_i = np.argmax(prod*(1.0 - diff))
    best_threshold = confidences[best_i]

    return best_threshold, precision[best_i], recall[best_i]


def compute_iou(box1, box2):
    w = min(box1['xmax'], box2['xmax']) - max(box1['xmin'], box2['xmin'])
    if w > 0:
        h = min(box1['ymax'], box2['ymax']) - max(box1['ymin'], box2['ymin'])
        if h > 0:
            intersection = w*h
            w1 = box1['xmax'] - box1['xmin']
            h1 = box1['ymax'] - box1['ymin']
            w2 = box2['xmax'] - box2['xmin']
            h2 = box2['ymax'] - box2['ymin']
            union = (w1*h1 + w2*h2) - intersection
            return float(intersection)/float(union)
    return 0.0


def match(detection, groundtruth_boxes):
    """
    Arguments:
        detection: a box.
        groundtruth_boxes: a list of boxes.
    Returns:
        best_i: an integer, index of the best groundtruth box.
        max_iou: a float number.
    """
    best_i = -1
    max_iou = 0.0
    for i, box in enumerate(groundtruth_boxes):
        iou = compute_iou(detection, box)
        if iou > max_iou:
            best_i = i
            max_iou = iou
    return best_i, max_iou


def compute_ap(precision, recall):
    previous_recall_value = 0.0
    ap = 0.0
    # recall is in increasing order
    for p, r in zip(precision, recall):
        delta = r - previous_recall_value
        ap += p*delta
        previous_recall_value = r
    return ap
