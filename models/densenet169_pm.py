import os
import sys

import tensorflow as tf
import scipy
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))

import tf_util
import pointnet
from custom_layers import Scale
from keras.layers import (Input, Dense, Convolution2D, MaxPooling2D, 
                          AveragePooling2D, GlobalAveragePooling2D, 
                          ZeroPadding2D, Dropout, Flatten, add, 
                          concatenate, Reshape, Activation)
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K
K.set_learning_phase(1) #set learning phase

def placeholder_inputs(batch_size, img_rows=224, img_cols=224, points=16384, separately=False):
    imgs_pl = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, 3))
    fmaps_pl = tf.placeholder(tf.float32, shape=(batch_size, img_rows, img_cols, 3))    
    if separately:
        speeds_pl = tf.placeholder(tf.float32, shape=(batch_size))
        angles_pl = tf.placeholder(tf.float32, shape=(batch_size))
        labels_pl = [speeds_pl, angles_pl]
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, 2))
    return imgs_pl, fmaps_pl, labels_pl


def get_densenet(img_rows, img_cols, nb_dense_block=4, 
                 growth_rate=32, nb_filter=64, reduction=0.5, 
                 dropout_rate=0.0, weight_decay=1e-4):
    '''
    DenseNet 169 Model for Keras

    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfN0d3T1F1MXg0NlU
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSEc5UC1ROUFJdmM

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    img_input = Input(shape=(224, 224, 3), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,32,32] # For DenseNet-169

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=3, name='conv1_bn')(x)
    x = Scale(axis=3, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=3, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=3, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    # Use pre-trained weights for Tensorflow backend
    weights_path = 'utils/weights/densenet169_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    x_newfc = Dense(256, name='fc7')(x_newfc)
    model = Model(img_input, x_newfc)

    return model


def get_model(net, is_training, add_lstm=True, bn_decay=None, separately=False):
    """ Densenet169 regression model, input is BxWxHx3, output Bx2"""
    batch_size = net[0].get_shape()[0].value
    img_net, fmap_net = net[0], net[1]

    img_net = get_densenet(224, 224)(img_net)
    fmap_net = get_densenet(224, 224)(fmap_net)

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


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=3, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, (1, 1), name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=3, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, (3, 3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base+'_bn')(x)
    x = Scale(axis=3, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=3, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


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
