import tensorflow as tf
from .rpn import rpn
from .head import head
from .utils import decode, batch_decode, batch_multiclass_non_max_suppression
from .training_target_creation import get_training_targets
from .losses_and_subsampling import positive_negative_subsample,\
    localization_loss, classification_loss


class Detector:
    def __init__(self, images, feature_extractor, is_training, params):
        """
        Arguments:
            images: a float tensor with shape [batch_size, image_height, image_width, 3],
                it represents RGB images with pixel values in range [0, 1].
            feature_extractor: it takes images and returns a dict with two float tensors.
            is_training: a boolean.
            params: a dict.
        """

        # image size is dynamic
        image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]
        image_size = (image_width, image_height)

        # the main computational graph is build here
        features = feature_extractor(images)
        self.proposals, self.rpn_output, self.anchors = rpn(
            features['rpn_features'], is_training,
            image_size, params
        )
        self.encoded_boxes, self.logits = head(
            features['second_stage_features'],
            self.proposals['rois'],
            self.proposals['roi_image_indices'],
            image_size, params
        )

    def get_predictions(self, score_threshold=0.1, iou_threshold=0.4, max_boxes_per_class=100):
        """Postprocess outputs of the network.
        Returns:
            boxes: a float tensor with shape [N, 4],
                where 0 <= N <= num_classes * max_boxes_per_class * batch_size.
            labels: an int tensor with shape [N].
            scores: a float tensor with shape [N].
            num_boxes_per_image: an int tensor with shape [batch_size], it
                represents the number of detections on an image.
        """
        with tf.name_scope('decoding'):

            rois = tf.concat(self.proposals['rois'], axis=0)
            # shape [total_num_proposals, 4]

            encoded_boxes = tf.transpose(self.encoded_boxes, perm=[1, 0, 2])
            # shape [num_classes, total_num_proposals, 4]

            boxes = batch_decode(encoded_boxes, rois)
            boxes = tf.transpose(boxes, perm=[1, 0, 2])
            # it has shape [total_num_proposals, num_classes, 4]

            probabilities = tf.nn.softmax(self.logits, axis=1)[:, 1:]
            # it has shape [total_num_proposals, num_classes]

        with tf.name_scope('nms'):
            boxes, scores, classes, num_boxes = batch_multiclass_non_max_suppression(
                boxes, probabilities, self.proposals['num_proposals_per_image'],
                score_threshold, iou_threshold,
                max_boxes_per_class
            )
        predictions = {
            'boxes': boxes, 'labels': classes, 'scores': scores,
            'num_boxes_per_image': num_boxes
        }
        return predictions

    def get_losses(self, groundtruth, params):
        """
        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, max_num_boxes, 4].
                'labels': an int tensor with shape [batch_size, max_num_boxes].
                'num_boxes': an int tensor with shape [batch_size],
                    where `max_num_boxes = max(num_boxes)`.
            params: a dict.
        Returns:
            a dict with four float tensors with shape [].
        """

        groundtruth_boxes_list = tf.unstack(groundtruth['boxes'], axis=0)
        groundtruth_labels_list = tf.unstack(groundtruth['labels'], axis=0)
        num_groundtruth_boxes_list = tf.unstack(groundtruth['num_boxes'], axis=0)

        rpn_encoded_boxes_list = tf.unstack(self.rpn_output['encoded_boxes'], axis=0)
        rpn_objectness_scores_list = tf.unstack(self.rpn_output['objectness_scores'], axis=0)

        rois_list = self.proposals['rois']
        num_proposals = self.proposals['num_proposals_per_image']  # shape [batch_size]
        batch_size = num_proposals.shape[0].value

        encoded_boxes_list = tf.split(self.encoded_boxes, num_or_size_splits=num_proposals, axis=0)
        logits_list = tf.split(self.logits, num_or_size_splits=num_proposals, axis=0)

        # all losses will be collected here
        losses = {
            'roi_localization_loss': [], 'roi_classification_loss': [],
            'rpn_localization_loss': [], 'rpn_classification_loss': []
        }

        # compute losses for each image in the batch
        for i in range(batch_size):

            # get groundtruth data for the image
            N = num_groundtruth_boxes_list[i]
            groundtruth_boxes = groundtruth_boxes_list[i][:N]  # shape [N, 4]
            groundtruth_labels = groundtruth_labels_list[i][:N]  # shape [N]

            # raw outputs of the region proposal network
            rpn_encoded_boxes = rpn_encoded_boxes_list[i]  # shape [num_anchors, 4]
            rpn_objectness_scores = rpn_objectness_scores_list[i]  # shape [num_anchors, 2]

            # get proposals for the image
            proposals = rois_list[i]  # shape [num_proposals_i, 4]

            # get final predictions for each proposal on the image
            encoded_boxes = encoded_boxes_list[i]  # shape [num_proposals_i, num_classes, 4]
            logits = logits_list[i]  # shape [num_proposals_i, num_classes + 1]

            targets, anchor_matches, proposal_matches = get_training_targets(
                self.anchors, proposals,
                groundtruth_boxes, groundtruth_labels,
                positives_threshold=params['positives_threshold'],
                negatives_threshold=params['negatives_threshold'],
                second_stage_threshold=params['second_stage_threshold']
            )

            with tf.name_scope('rpn_losses_for_image_%d' % i):

                # whether an anchor is matched
                is_positive_anchor = tf.greater_equal(anchor_matches, 0)

                # whether i can use an anchor when computing loss
                to_not_ignore = tf.not_equal(anchor_matches, -2)

                is_chosen_anchor = positive_negative_subsample(
                    to_not_ignore, is_positive_anchor,
                    batch_size=params['first_stage_batch_size'],
                    positive_fraction=params['positive_fraction']
                )  # shape [num_anchors]
                is_chosen_anchor_positive = tf.logical_and(
                    is_chosen_anchor, is_positive_anchor
                )

                rpn_loc_losses = localization_loss(
                    rpn_encoded_boxes, targets['rpn_regression'],
                    weights=tf.to_float(is_chosen_anchor_positive)
                )
                rpn_cls_losses = classification_loss(
                    rpn_objectness_scores, tf.to_int32(is_chosen_anchor_positive),
                    weights=tf.to_float(is_chosen_anchor)
                )
                # they have shape [first_stage_batch_size]

            with tf.name_scope('second_stage_losses_for_image_%d' % i):

                encoded_boxes = tf.pad(encoded_boxes, [[0, 0], [1, 0], [0, 0]])
                # now it has shape [num_proposals_i, num_classes + 1, 4]

                class_indices = targets['roi_classification']  # shape [num_proposals_i]
                proposal_indices = tf.range(tf.size(class_indices), dtype=tf.int32)
                indices = tf.stack([proposal_indices, class_indices], axis=1)  # shape [num_proposals_i, 2]
                encoded_boxes = tf.gather_nd(encoded_boxes, indices)  # shape [num_proposals_i, 4]
                # i did this because different classes don't share boxes

                # whether proposal is matched
                is_roi_positive = tf.to_float(tf.greater_equal(proposal_matches, 0))

                loc_losses = localization_loss(
                    encoded_boxes, targets['roi_regression'],
                    weights=is_roi_positive
                )
                cls_losses = classification_loss(
                    logits, targets['roi_classification'],
                    weights=None
                )
                # they have shape [num_proposals_i]

            with tf.name_scope('ohem'):

                # i don't need to do nms here,
                # because i did nms at proposal generation stage
                k = tf.minimum(params['num_hard_examples'], tf.shape(cls_losses)[0])
                _, hard_rois_indices = tf.nn.top_k(2.0 * loc_losses + cls_losses, k, sorted=False)
                hard_loc_losses = tf.gather(loc_losses, hard_rois_indices)  # shape [k]
                hard_cls_losses = tf.gather(cls_losses, hard_rois_indices)  # shape [k]

            with tf.name_scope('normalization'):

                normalizer = tf.maximum(tf.reduce_sum(tf.to_float(is_chosen_anchor), axis=0), 1.0)
                rpn_loc_loss = tf.reduce_sum(rpn_loc_losses, axis=0) / normalizer
                rpn_cls_loss = tf.reduce_sum(rpn_cls_losses, axis=0) / normalizer

                normalizer = tf.maximum(tf.to_float(k), 1.0)  # k can be zero
                hard_loc_loss = tf.reduce_sum(hard_loc_losses, axis=0) / normalizer
                hard_cls_loss = tf.reduce_sum(hard_cls_losses, axis=0) / normalizer

            tf.summary.scalar('num_groundtruth_boxes', tf.to_float(N))
            tf.summary.scalar('num_rpn_positives', tf.reduce_sum(tf.to_float(is_positive_anchor), axis=0))
            tf.summary.scalar('num_roi_positives', tf.reduce_sum(is_roi_positive, axis=0))
            num_hard_roi_positives = tf.reduce_sum(tf.gather(is_roi_positive, hard_rois_indices), axis=0)
            tf.summary.scalar('num_hard_roi_positives', num_hard_roi_positives)

            losses['roi_localization_loss'].append(hard_loc_loss)
            losses['roi_classification_loss'].append(hard_cls_loss)
            losses['rpn_localization_loss'].append(rpn_loc_loss)
            losses['rpn_classification_loss'].append(rpn_cls_loss)

        # take means over the batch
        losses = {n: tf.add_n(v)/batch_size for n, v in losses.items()}
        return losses
