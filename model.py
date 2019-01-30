import tensorflow as tf
from detector import Detector
from detector.backbones import resnet, mobilenet, shufflenet
from metrics import Evaluator


MOVING_AVERAGE_DECAY = 0.997


def model_fn(features, labels, mode, params, config):
    """
    This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """

    # choose a backbone network
    if params['backbone'] == 'resnet':
        feature_extractor = resnet
    elif params['backbone'] == 'mobilenet':
        feature_extractor = lambda x: mobilenet(x, params['depth_multiplier'])
    elif params['backbone'] == 'shufflenet':
        feature_extractor = lambda x: shufflenet(x, str(params['depth_multiplier']))

    # build the main graph
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    detector = Detector(features['images'], feature_extractor, is_training, params)

    # add NMS to the graph
    if not is_training:
        predictions = detector.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_boxes_per_class=params['max_boxes_per_class']
        )

    if mode == tf.estimator.ModeKeys.PREDICT:

        w, h = tf.unstack(tf.to_float(features['images_size']))  # original image size
        s = tf.to_float(tf.shape(features['images']))  # size after resizing
        scaler = tf.stack([h/s[1], w/s[2], h/s[1], w/s[2]])
        predictions['boxes'] = scaler * predictions['boxes']

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
            evaluator = Evaluator(num_classes=params['num_classes'])
            eval_metric_ops = evaluator.get_metric_ops(labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.cosine_decay(
            params['initial_learning_rate'],
            global_step, decay_steps=params['num_steps']
        )
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

        if params['backbone'] == 'shufflenet':
            var_list = [
                v for v in tf.trainable_variables()
                if 'Conv1' not in v.name and 'Stage2' not in v.name
            ]
        elif params['backbone'] == 'mobilenet':
            var_list = [
                v for v in tf.trainable_variables()
                if all('Conv2d_%d_' % i not in v.name for i in range(6))
                and 'Conv2d_0' not in v.name
            ]
        elif params['backbone'] == 'resnet':
            var_list = [
                v for v in tf.trainable_variables()
                if 'resnet_v1_50/block1/' not in v.name
                and 'resnet_v1_50/conv1/' not in v.name
            ]

        grads_and_vars = optimizer.compute_gradients(total_loss, var_list)
        grads_and_vars = [
            (3.0 * g, v) if 'thin_feature_maps' in v.name else (g, v)
            for g, v in grads_and_vars
        ]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [
        v for v in tf.trainable_variables()
        if 'weights' in v.name and 'depthwise_weights' not in v.name
    ]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)


class RestoreMovingAverageHook(tf.train.SessionRunHook):
    def __init__(self, model_dir):
        super(RestoreMovingAverageHook, self).__init__()
        self.model_dir = model_dir

    def begin(self):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)
        variables_to_restore = ema.variables_to_restore()
        self.load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(self.model_dir), variables_to_restore
        )

    def after_create_session(self, sess, coord):
        tf.logging.info('Loading EMA weights...')
        self.load_ema(sess)
