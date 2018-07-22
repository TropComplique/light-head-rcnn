import tensorflow as tf


def localization_loss(predictions, targets, weights):
    """A usual L1 smooth loss.
    Arguments:
        predictions: a float tensor with shape [N, 4].
        targets: a float tensor with shape [N, 4].
        weights: a float tensor with shape [N].
    Returns:
        a float tensor with shape [N].
    """
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    losses = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    return weights * tf.reduce_sum(losses, axis=1)


def classification_loss(predictions, targets, weights=None):
    """
    Arguments:
        predictions: a float tensor with shape [N, num_classes + 1].
        targets: an int tensor with shape [N].
        weights: a float tensor with shape [N] or None.
    Returns:
        a float tensor with shape [N].
    """
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=predictions)
    return losses if weights is None else weights * losses


def positive_negative_subsample(indicator, is_positive, batch_size=256, positive_fraction=0.5):
    """
    In the case of training RPN:
        indicator - non ignored anchors indicator,
        is_positive - matched anchors indicator,
        batch_size - number of anchors to use when computing loss,
        positive_fraction - fraction of positive anchors in created batch of anchors.

    See section "Training RPNs" in the original faster-rcnn paper.

    Arguments:
        indicator: a boolean tensor with shape [N] whose `True` entries can be sampled.
        is_positive: a boolean tensor with shape [N] denoting positive and negative examples.
        batch_size: an integer.
        positive_fraction: a float number, desired fraction of positive
            examples in the batch. It must have value in range [0, 1].
    Returns:
        a boolean tensor with shape [N], `True` for entries which are sampled.
    """
    # only sample from indicated samples
    is_negative = tf.logical_and(tf.logical_not(is_positive), indicator)
    is_positive = tf.logical_and(is_positive, indicator)

    # sample positives
    max_num_positives = tf.constant(int(positive_fraction * batch_size), dtype=tf.int32)
    sampled_positives_indicator = subsample_indicator(is_positive, max_num_positives)

    # sample negatives
    max_num_negatives = batch_size - tf.reduce_sum(tf.to_int32(sampled_positives_indicator))
    sampled_negatives_indicator = subsample_indicator(is_negative, max_num_negatives)

    is_sampled = tf.logical_or(sampled_positives_indicator, sampled_negatives_indicator)
    # if sum(indicator) >= batch_size
    # then sum(is_sampled) = batch_size,
    # if sum(indicator) < batch_size
    # then sum(is_sampled) = sum(indicator)
    return is_sampled


def subsample_indicator(indicator, num_samples):
    """
    Given a boolean indicator vector with M elements set to `True`, the function
    assigns all but `num_samples` of these previously `True` elements to `False`.
    If `num_samples` is greater than M, the original indicator vector is returned.

    Arguments:
        indicator: a boolean tensor with shape [N] indicating which elements
            are allowed to be sampled and which are not.
        num_samples: an int tensor with shape [].
    Returns:
        a boolean tensor with shape [N].
    """
    indices = tf.squeeze(tf.where(indicator), axis=1)  # shape [M]
    indices = tf.random_shuffle(indices)

    num_samples = tf.minimum(tf.size(indices), num_samples)
    selected_indices = tf.slice(indices, [0], [num_samples])

    selected_indicator = tf.sparse_to_dense(
        sparse_indices=tf.to_int32(selected_indices),
        output_shape=tf.shape(indicator),
        sparse_values=1, default_value=0,
        validate_indices=False
    )

    # if sum(indicator) >= num_samples
    # then sum(selected_indicator) = num_samples,
    # if sum(indicator) < num_samples
    # then selected_indicator = indicator
    return tf.equal(selected_indicator, 1)
