import tensorflow as tf
from detector.utils import encode, iou


def get_training_targets(
        anchors, proposals, groundtruth_boxes, groundtruth_labels,
        positives_threshold=0.7, negatives_threshold=0.3,
        second_stage_threshold=0.5):
    """
    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        proposals: a float tensor with shape [num_proposals, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        groundtruth_labels: an int tensor with shape [N],
            with values in range [0, num_classes).
        positives_threshold: a float number.
        negatives_threshold: a float number.
        second_stage_threshold: a float number.
    Returns:
        rpn_regression: a float tensor with shape [num_anchors, 4].
        roi_regression: a float tensor with shape [num_proposals, 4].
        roi_classification: an int tensor with shape [num_proposals].
        proposal_matches: an int tensor with shape [num_proposals].
        anchor_matches: an int tensor with shape [num_anchors].
    """
    N = tf.shape(groundtruth_boxes)[0]

    with tf.name_scope('anchor_matching'):
        num_anchors = tf.shape(anchors)[0]
        only_background = tf.fill([num_anchors], -1)
        matches = tf.cond(
            tf.greater(N, 0),
            lambda: match_boxes(
                anchors, groundtruth_boxes,
                positives_threshold=positives_threshold,
                negatives_threshold=negatives_threshold,
                force_match_groundtruth=True
            ),
            lambda: only_background
        )
        anchor_matches = tf.to_int32(matches)

    with tf.name_scope('proposal_matching'):
        num_proposals = tf.shape(proposals)[0]
        only_background = tf.fill([num_proposals], -1)
        matches = tf.cond(
            tf.greater(N * num_proposals, 0),
            lambda: match_boxes(
                proposals, groundtruth_boxes,
                positives_threshold=second_stage_threshold,
                negatives_threshold=second_stage_threshold,
                force_match_groundtruth=False
            ),
            lambda: only_background
        )
        proposal_matches = tf.to_int32(matches)

    with tf.name_scope('rpn_target_creation'):
        rpn_reg_targets, _ = create_targets(
            anchors, groundtruth_boxes,
            groundtruth_labels, anchor_matches
        )
        # we don't need classification targets for rpn,
        # we only need to know which anchors are positive/negative

    with tf.name_scope('second_stage_target_creation'):
        reg_targets, cls_targets = create_targets(
            proposals, groundtruth_boxes,
            groundtruth_labels, proposal_matches
        )

    targets = {
        'rpn_regression': rpn_reg_targets,
        'roi_regression': reg_targets,
        'roi_classification': cls_targets
    }
    return targets, anchor_matches, proposal_matches


def match_boxes(
        anchors, groundtruth_boxes, positives_threshold=0.7,
        negatives_threshold=0.3, force_match_groundtruth=True):
    """
    Note: this function can also be used to match proposals to groundtruth,
    in this case
    1. anchors -> proposals
    2. positives_threshold = negatives_threshold = 0.5
    3. force_match_groundtruth = False
    Below is described anchor matching, not proposal matching.

    If the anchor has IoU over `positives_threshold` with any groundtruth box,
    it will be set a positive label.
    Anchors which have highest IoU for a groundtruth box will
    also be assigned a positive label.
    Meanwhile, if other anchors have IoU less than `negatives_threshold`
    with all groundtruth boxes, their labels will be negative.

    Matching algorithm:
    1) for each groundtruth box choose the anchor with largest IoU,
    2) remove this set of anchors from the set of all anchors,
    3) for each remaining anchor choose the groundtruth box with largest IoU,
       but only if this IoU is larger than `positives_threshold`,
    4) remove this set of matched anchors from the set of all anchors,
    5) for each remaining anchor if it has IoU less than `negatives_threshold`
       with all groundtruth boxes set it to `negative`, otherwise set it to `ignore`.

    Note: after step 1, it could happen that for some two groundtruth boxes
    chosen anchors are the same. Let's hope this never happens.
    Also see the comments below.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        positives_threshold: a float number.
        negatives_threshold: a float number.
        force_match_groundtruth: a boolean, whether to try to make sure
            that all groundtruth boxes are matched.
    Returns:
        an int tensor with shape [num_anchors], possible values
            that it can contain are [-2, -1, 0, 1, 2, ..., (N - 1)],
            where numbers in the range [0, N - 1] mean indices of the groundtruth boxes,
            `-1` means that an anchor box is negative (background),
            and `-2` means that we must ignore this anchor box.
    """
    assert positives_threshold >= negatives_threshold

    # for each anchor box choose the groundtruth box with largest iou
    similarity_matrix = iou(groundtruth_boxes, anchors)  # shape [N, num_anchors]
    matches = tf.argmax(similarity_matrix, axis=0, output_type=tf.int32)  # shape [num_anchors]
    matched_vals = tf.reduce_max(similarity_matrix, axis=0)  # shape [num_anchors]
    is_positive = tf.to_int32(tf.greater_equal(matched_vals, positives_threshold))

    if positives_threshold == negatives_threshold:
        is_negative = 1 - is_positive
        matches = matches * is_positive + (-1 * is_negative)
    else:
        is_negative = tf.to_int32(tf.greater(negatives_threshold, matched_vals))
        to_ignore = (1 - is_positive) * (1 - is_negative)
        matches = matches * is_positive + (-1 * is_negative) + (-2 * to_ignore)

    # after this, it could happen that some groundtruth
    # boxes are not matched with any anchor box

    if force_match_groundtruth:
        # now we must ensure that each row (groundtruth box) is matched to
        # at least one column (which is not guaranteed
        # otherwise if `positives_threshold` is high)

        # for each groundtruth box choose the anchor box with largest iou
        # (force match for each groundtruth box)
        forced_matches_ids = tf.argmax(similarity_matrix, axis=1, output_type=tf.int32)  # shape [N]
        # if all indices in forced_matches_ids are different then all rows will be matched

        num_anchors = tf.shape(anchors)[0]
        forced_matches_indicators = tf.one_hot(forced_matches_ids, depth=num_anchors, dtype=tf.int32)  # shape [N, num_anchors]
        forced_match_row_ids = tf.argmax(forced_matches_indicators, axis=0, output_type=tf.int32)  # shape [num_anchors]

        # some forced matches could be very bad!
        forced_matches_values = tf.reduce_max(similarity_matrix, axis=1)  # shape [N]
        small_iou = 0.05  # this requires that forced match has at least small intersection
        is_okay = tf.to_int32(tf.greater_equal(forced_matches_values, small_iou))  # shape [N]
        forced_matches_indicators = forced_matches_indicators * tf.expand_dims(is_okay, axis=1)

        forced_match_mask = tf.greater(tf.reduce_max(forced_matches_indicators, axis=0), 0)  # shape [num_anchors]
        matches = tf.where(forced_match_mask, forced_match_row_ids, matches)
        # even after this it could happen that some rows aren't matched,
        # but i believe that this event has low probability

    return matches


def create_targets(anchors, groundtruth_boxes, groundtruth_labels, matches):
    """Returns regression and classification targets for each anchor.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        groundtruth_labels: an int tensor with shape [N], possible values
            that it can contain are [0, 1, 2, ..., (num_classes - 1)].
        matches: an int tensor with shape [num_anchors].
        num_classes: an integer.
    Returns:
        reg_targets: a float tensor with shape [num_anchors, 4].
        cls_targets: an int tensor with shape [num_anchors], possible values
            that it can contain are [0, 1, 2, ..., num_classes],
            `0` is a background class.
    """
    matched_anchor_indices = tf.where(tf.greater_equal(matches, 0))  # shape [num_matches, 1]
    matched_anchor_indices = tf.to_int32(tf.squeeze(matched_anchor_indices, axis=1))

    unmatched_anchor_indices = tf.where(tf.less(matches, 0))  # shape [num_anchors - num_matches, 1]
    unmatched_anchor_indices = tf.to_int32(tf.squeeze(unmatched_anchor_indices, axis=1))

    matched_gt_indices = tf.gather(matches, matched_anchor_indices)  # shape [num_matches]
    matched_gt_boxes = tf.gather(groundtruth_boxes, matched_gt_indices)  # shape [num_matches, 4]
    matched_anchors = tf.gather(anchors, matched_anchor_indices)  # shape [num_matches, 4]

    matched_reg_targets = encode(matched_gt_boxes, matched_anchors)  # shape [num_matches, 4]
    matched_cls_targets = tf.gather(groundtruth_labels, matched_gt_indices)  # shape [num_matches]
    matched_cls_targets = matched_cls_targets + 1  # background class will have index `0`

    num_unmatched = tf.size(unmatched_anchor_indices)  # num_anchors - num_matches
    unmatched_reg_targets = tf.zeros([num_unmatched, 4], dtype=tf.float32)
    unmatched_cls_targets = tf.zeros([num_unmatched], dtype=tf.int32)

    reg_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_reg_targets, unmatched_reg_targets]
    )  # shape [num_anchors, 4]

    cls_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_cls_targets, unmatched_cls_targets]
    )  # shape [num_anchors]

    return reg_targets, cls_targets
