import tensorflow as tf


def multiclass_non_max_suppression(
        boxes, scores, score_threshold,
        iou_threshold, max_boxes_per_class):
    """Multi-class version of non maximum suppression. It operates independently
    for each class. Also it prunes boxes with score less than a provided
    threshold prior to applying NMS.

    Arguments:
        boxes: a float tensor with shape [N, num_classes, 4].
        scores: a float tensor with shape [N, num_classes].
        score_threshold: a float number.
        iou_threshold: a float number.
        max_boxes_per_class: an integer,
            maximum number of retained boxes per class.
    Returns:
        selected_boxes: a float tensor with shape [M, 4],
            where 0 <= M <= N.
        selected_scores: a float tensor with shape [M].
        selected_classes: an int tensor with shape [M].        .
    """
    boxes_list = tf.unstack(boxes, axis=1)
    scores_list = tf.unstack(scores, axis=1)

    selected_boxes, selected_scores, selected_classes = [], [], []
    for i, (b, s) in enumerate(boxes_list, scores_list):

        selected_indices = tf.image.non_max_suppression(
            boxes=b, scores=s, max_output_size=max_boxes_per_class,
            iou_threshold=iou_threshold, score_threshold=score_threshold,
        )
        selected_boxes += [tf.gather(class_boxes, selected_indices)]
        selected_scores += [tf.gather(class_scores, selected_indices)]
        selected_classes += [i * tf.ones_like(selected_indices)]

    selected_boxes = tf.concat(selected_boxes, axis=0)
    selected_scores = tf.concat(selected_scores, axis=0)
    selected_classes = tf.to_int32(tf.concat(selected_classes, axis=0))
    return selected_boxes, selected_scores, selected_classes


def batch_multiclass_non_max_suppression(
        boxes, scores, num_boxes_per_image,
        score_threshold, iou_threshold,
        max_boxes_per_class):
    """Same as multiclass_non_max_suppression but for a batch of images.

    Arguments:
        boxes: a float tensor with shape [N, num_classes, 4].
        scores: a float tensor with shape [N, num_classes].
        num_boxes_per_image: an int tensor with shape [batch_size],
            where N = sum(num_boxes_per_image).
    Returns:
        boxes: a float tensor with shape [M, 4].
        scores: a float tensor with shape [M].
        classes: an int tensor with shape [M].
        num_boxes_per_image: an int tensor with shape [batch_size].

    """
    batch_size = num_boxes_per_image.shape[0].value
    boxes_list = tf.split(boxes, num_or_size_splits=num_boxes_per_image, axis=0)
    scores_list = tf.split(scores, num_or_size_splits=num_boxes_per_image, axis=0)

    selected_boxes, selected_scores, selected_classes = [], [], []
    num_selected_boxes_per_image = []

    for i in range(batch_size):

        b, s, c = multiclass_non_max_suppression(
            boxes_list[i], scores_list[i],
            score_threshold, iou_threshold,
            max_boxes_per_class
        )
        n = tf.to_int32(tf.shape(b)[0])

        selected_boxes.append(b)
        selected_scores.append(s)
        selected_classes.append(c)
        num_selected_boxes_per_image.append(n)

    boxes = tf.concat(selected_boxes, axis=0)
    scores = tf.concat(selected_scores, axis=0)
    classes = tf.concat(selected_classes, axis=0)
    num_boxes_per_image = tf.stack(num_selected_boxes_per_image)

    return boxes, scores, classes, num_boxes_per_image
