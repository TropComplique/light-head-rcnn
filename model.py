import tensorflow as tf
from detector import Detector
from detector.backbones import resnet
from metrics import Evaluator


def model_fn(features, labels, mode, params, config):
    """This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """

    # build the main graph
    feature_extractor = resnet
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    detector = Detector(features['images'], feature_extractor, is_training, params)

    # use a pretrained backbone network
    if is_training:
        with tf.name_scope('init_from_checkpoint'):
            tf.train.init_from_checkpoint(params['pretrained_checkpoint'], {'resnet_v1_50/': 'resnet_v1_50/'})

    # add NMS to the graph
    if not is_training:
        predictions = detector.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_boxes_per_class=params['max_boxes_per_class']
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        # this is required for exporting a savedmodel
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    # add L2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    # create localization and classification losses
    losses = detector.get_losses(labels, params)
    tf.losses.add_loss(params['alpha'] * losses['rpn_localization_loss'])
    tf.losses.add_loss(params['beta'] * losses['rpn_classification_loss'])
    tf.losses.add_loss(params['gamma'] * losses['roi_localization_loss'])
    tf.losses.add_loss(params['theta'] * losses['roi_classification_loss'])
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('rpn_localization_loss', losses['rpn_localization_loss'])
    tf.summary.scalar('rpn_classification_loss', losses['rpn_classification_loss'])
    tf.summary.scalar('roi_localization_loss', losses['roi_localization_loss'])
    tf.summary.scalar('roi_classification_loss', losses['roi_classification_loss'])

    if mode == tf.estimator.ModeKeys.EVAL:

        with tf.name_scope('evaluator'):
            evaluator = Evaluator(class_names=['face'])
            eval_metric_ops = evaluator.get_metric_ops(
                features['filenames'], labels, predictions
            )

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [v for v in tf.trainable_variables() if 'weights' in v.name]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)
