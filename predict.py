import argparse
import importlib
import os
import sys
import time

import numpy as np
import scipy

import provider
import tensorflow as tf

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='nvidia_pn',
                    help='Model name [default: nvidia_pn]')
parser.add_argument('--model_path', default='logs/nvidia_pn/model.ckpt',
                    help='Model checkpoint file path [default: logs/nvidia_pn/model.ckpt]')
parser.add_argument('--max_epoch', type=int, default=250,
                    help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch Size during training [default: 8]')
parser.add_argument('--result_dir', default='results',
                    help='Result folder path [results]')

FLAGS = parser.parse_args()
BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path

assert (FLAGS.model == "nvidia_pn")
MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

RESULT_DIR = os.path.join(FLAGS.result_dir, FLAGS.model)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
LOG_FOUT = open(os.path.join(RESULT_DIR, 'log_predict.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def predict():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        if 'pn' in MODEL_FILE:
            data_input = provider.Provider()
            imgs_pl, pts_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE)
            imgs_pl = [imgs_pl, pts_pl]
        else:
            raise NotImplementedError

        is_training_pl = tf.placeholder(tf.bool, shape=())
        print(is_training_pl)

        # Get model and loss
        pred = MODEL.get_model(imgs_pl, is_training_pl)

        loss = MODEL.get_loss(pred, labels_pl)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'imgs_pl': imgs_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'loss': loss}

    pred_one_epoch(sess, ops, data_input)

def pred_one_epoch(sess, ops, data_input):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    preds = []
    num_batches = data_input.num_test // BATCH_SIZE

    for batch_idx in range(num_batches):
        if "io" in MODEL_FILE:
            imgs = data_input.load_one_batch(BATCH_SIZE, "test")
            feed_dict = {ops['imgs_pl']: imgs,
                         ops['is_training_pl']: is_training}
        else:
            imgs, others = data_input.load_one_batch(BATCH_SIZE, "test")
            feed_dict = {ops['imgs_pl'][0]: imgs,
                         ops['imgs_pl'][1]: others,
                         ops['is_training_pl']: is_training}

        pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
        preds.append(pred_val)

    preds = np.vstack(preds)
    print (preds.shape)
    # preds[:, 1] = preds[:, 1] * 180.0 / scipy.pi
    # preds[:, 0] = preds[:, 0] * 20 + 20

    np.savetxt(os.path.join(RESULT_DIR, "behavior_pred.txt"), preds)

    output_dir = os.path.join(RESULT_DIR, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i_list = get_dicts(description="test")
    counter = 0
    for i, num in enumerate(i_list):
        np.savetxt(os.path.join(output_dir, str(i) + ".txt"), preds[counter:counter+num,:])
        counter += num
    # plot_acc(preds, labels)

def get_dicts(description="val"):
    if description == "train":
        raise NotImplementedError
    elif description == "val": # batch_size == 8
        return [120] * 4 + [111] + [120] * 4 + [109] + [120] * 9 + [89 - 87 % 8]
    elif description == "test": # batch_size == 8
        return [120] * 9 + [116] + [120] * 4 + [106] + [120] * 4 + [114 - 114 % 8]
    else:
        raise NotImplementedError

if __name__ == "__main__":
    predict()
    # plot_acc_from_txt()
