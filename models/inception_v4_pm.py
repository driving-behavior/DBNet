import os
import sys

import tensorflow as tf
import scipy
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tf_util
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, add, concatenate, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
K.set_learning_phase(1) #set learning phase


def placeholder_inputs(batch_size, img_rows=299, img_cols=299, points=16384, separately=False):
    imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, 3))
    fmaps_pl = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, 3))    
    if separately:
        speeds_pl = tf.placeholder(tf.float32, shape=(batch_size))
        angles_pl = tf.placeholder(tf.float32, shape=(batch_size))
        labels_pl = [speeds_pl, angles_pl]
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 2))
    return imgs_pl, fmaps_pl, labels_pl


def get_inception(img_rows=299, img_cols=299, dropout_keep_prob=0.2, separately=False):
    '''
    Inception V4 Model for Keras

    Model Schema is based on
    https://github.com/kentsommer/keras-inceptionV4

    ImageNet Pretrained Weights
    Theano: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_th_dim_ordering_th_kernels.h5
    TensorFlow: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    '''

    # Input Shape is 299 x 299 x 3 (tf)
    img_input = Input(shape=(img_rows, img_cols, 3), name='data')

    # Make inception base
    net = inception_v4_base(img_input)

    # Final pooling and prediction

    # 8 x 8 x 1536
    net_old = AveragePooling2D((8,8), padding='valid')(net)

    # 1 x 1 x 1536
    net_old = Dropout(dropout_keep_prob)(net_old)
    net_old = Flatten()(net_old)

    # 1536
    predictions = Dense(units=1001, activation='softmax')(net_old)

    model = Model(img_input, predictions, name='inception_v4')

    weights_path = 'utils/weights/inception-v4_weights_tf.h5'
    assert (os.path.exists(weights_path))
    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    net_ft = AveragePooling2D((8,8), border_mode='valid')(net)
    net_ft = Dropout(dropout_keep_prob)(net_ft)
    net_ft = Flatten()(net_ft)
    net = Dense(256, name='fc_mid')(net_ft)

    model = Model(img_input, net, name='inception_v4')
    return model


def get_model(net, is_training, add_lstm=True, bn_decay=None, separately=False):
    """ Inception_V4 regression model, input is BxWxHx3, output Bx2"""
    batch_size = net[0].get_shape()[0].value
    img_net, fmap_net = net[0], net[1]

    img_net = get_inception(299, 299)(img_net)
    fmap_net = get_inception(299, 299)(fmap_net)

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


def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), bias=False):
    """
    Utility function to apply conv + BN.
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    channel_axis = -1
    x = Convolution2D(nb_filter, (nb_row, nb_col),
                      strides=subsample,
                      padding=border_mode,
                      use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def block_inception_a(input):
    channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1)

    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_a(input):
    channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 3, 3, subsample=(2,2), border_mode='valid')

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, subsample=(2,2), border_mode='valid')

    branch_2 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_b(input):
    channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_b(input):
    channel_axis = -1

    branch_0 = conv2d_bn(input, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, subsample=(2,2), border_mode='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_c(input):
    channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1)

    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)


    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def inception_v4_base(input):
    channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = conv2d_bn(input, 32, 3, 3, subsample=(2,2), border_mode='valid')
    net = conv2d_bn(net, 32, 3, 3, border_mode='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)

    branch_1 = conv2d_bn(net, 96, 3, 3, subsample=(2,2), border_mode='valid')

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, border_mode='valid')

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, border_mode='valid')

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, subsample=(2,2), border_mode='valid')
    branch_1 = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in xrange(4):
      net = block_inception_a(net)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in xrange(7):
      net = block_inception_b(net)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in xrange(3):
      net = block_inception_c(net)

    return net


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
        new = scipy.misc.imresize(img, (299, 299))
        imgs_new.append(new)
    imgs_new = np.stack(imgs_new, axis=0)
    return imgs_new


if __name__ == '__main__':
    with tf.Graph().as_default():
        imgs = tf.zeros((32, 224, 224, 3))
        fmaps = tf.zeros((32, 224, 224, 3))
        outputs = get_model([imgs, fmaps], tf.constant(True))
        print(outputs)
        