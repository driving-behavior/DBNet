import os
import sys

import tensorflow as tf
import scipy
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tf_util
from custom_layers import Scale
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, add, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def placeholder_inputs(batch_size, img_rows=66, img_cols=200, separately=False):
    imgs_pl = tf.placeholder(tf.float32,
                             shape=(batch_size, img_rows, img_cols, 3))
    fmaps_pl = tf.placeholder(tf.float32,
                              shape=(batch_size, img_rows, img_cols, 3))
    if separately:
        speeds_pl = tf.placeholder(tf.float32, shape=(batch_size))
        angles_pl = tf.placeholder(tf.float32, shape=(batch_size))
        labels_pl = [speeds_pl, angles_pl]
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 2))
    return imgs_pl, fmaps_pl, labels_pl


def get_resnet(img_rows=224, img_cols=224, separately=False):
    """
    Resnet 152 Model for Keras

    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfZHhUT3lWVWxRN28/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
    """

    img_input = Input(shape=(img_rows, img_cols, 3), name='data')

    eps = 1.1e-5
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=3, name='bn_conv1')(x)
    x = Scale(axis=3, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    # Use pre-trained weights for Tensorflow backend
    weights_path = 'utils/weights/resnet152_weights_tf.h5'
    assert (os.path.exists(weights_path))

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(256, name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)
    return model


def get_model(net, is_training, add_lstm=True, bn_decay=None, separately=False):
    """ ResNet152 regression model, input is BxWxHx3, output Bx2"""
    batch_size = net[0].get_shape()[0].value
    img_net, fmap_net = net[0], net[1]

    img_net = get_resnet(224, 224)(img_net)
    fmap_net = get_resnet(224, 224)(fmap_net)

    net = tf.reshape(tf.stack([img_net, fmap_net]), [batch_size, -1])

    if not add_lstm:
        for i, dim in enumerate([256, 128, 16]):
            fc_scope = "fc" + str(i + 1)
            dp_scope = "dp" + str(i + 1)
            net = tf_util.fully_connected(net, dim, bn=True,
                                        is_training=is_training,
                                        scope=fc_scope,
                                        bn_decay=bn_decay)
            net = tf_util.dropout(net, keep_prob=0.7,
                                is_training=is_training,
                                scope=dp_scope)
        net = tf_util.fully_connected(net, 2, activation_fn=None, scope='fc4')
    else:
        fc_scope = "fc1"
        net = tf_util.fully_connected(net, 784, bn=True,
                                      is_training=is_training,
                                      scope=fc_scope,
                                      bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7,
                              is_training=is_training,
                              scope="dp1")
        net = cnn_lstm_block(net)
    return net


def cnn_lstm_block(input_tensor):
    lstm_in = tf.reshape(input_tensor, [-1, 28, 28])
    lstm_out = tf_util.stacked_lstm(lstm_in,
                                    num_outputs=10,
                                    time_steps=28,
                                    scope="cnn_lstm")

    W_final = tf.Variable(tf.truncated_normal([10, 2], stddev=0.1))
    b_final = tf.Variable(tf.truncated_normal([2], stddev=0.1))
    return tf.multiply(tf.atan(tf.matmul(lstm_out, W_final) + b_final), 2)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2a')(x)
    x = Scale(axis=3, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2b')(x)
    x = Scale(axis=3, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2c')(x)
    x = Scale(axis=3, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2a')(x)
    x = Scale(axis=3, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, (kernel_size, kernel_size),
                      name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2b')(x)
    x = Scale(axis=3, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2c')(x)
    x = Scale(axis=3, name=scale_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=3, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def get_loss(pred, label, l2_weight=0.0001):
    diff = tf.square(tf.subtract(pred, label))
    train_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars[1:]]) * l2_weight
    loss = tf.reduce_mean(diff + l2_loss)
    tf.summary.scalar('l2 loss', l2_loss * l2_weight)
    tf.summary.scalar('loss', loss)

    return loss


def summary_scalar(pred, label):
    threholds = [5, 4, 3, 2, 1, 0.5]
    angles = [float(t) / 180 * scipy.pi for t in threholds]
    speeds = [float(t) / 20 for t in threholds]

    for i in range(len(threholds)):
        scalar_angle = "angle(" + str(angles[i]) + ")"
        scalar_speed = "speed(" + str(speeds[i]) + ")"
        ac_angle = tf.abs(tf.subtract(pred[:, 1], label[:, 1])) < threholds[i]
        ac_speed = tf.abs(tf.subtract(pred[:, 0], label[:, 0])) < threholds[i]
        ac_angle = tf.reduce_mean(tf.cast(ac_angle, tf.float32))
        ac_speed = tf.reduce_mean(tf.cast(ac_speed, tf.float32))

        tf.summary.scalar(scalar_angle, ac_angle)
        tf.summary.scalar(scalar_speed, ac_speed)


def resize(imgs):
    batch_size = imgs.shape[0]
    imgs_new = []
    for j in range(batch_size):
        img = imgs[j,:,:,:]
        new = scipy.misc.imresize(img, (224, 224))
        imgs_new.append(new)
    imgs_new = np.stack(imgs_new, axis=0)
    return imgs_new


if __name__ == '__main__':
    with tf.Graph().as_default():
        imgs = tf.zeros((32, 224, 224, 3))
        fmaps = tf.zeros((32, 224, 224, 3))
        outputs = get_model([imgs, fmaps], tf.constant(True))
        print(outputs)
