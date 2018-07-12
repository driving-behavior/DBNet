import argparse
import importlib
import os
import sys
import time

# import matplotlib.pyplot as plt
import numpy as np
import scipy

import provider
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='nvidia_pn',
                    help='Model name [default: nvidia_pn]')
parser.add_argument('--model_path', default='logs/nvidia_pn/model_best.ckpt',
                    help='Model checkpoint file path [default: logs/nvidia_pn/model_best.ckpt]')
parser.add_argument('--max_epoch', type=int, default=250,
                    help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch Size during training [default: 8]')
parser.add_argument('--result_dir', default='results',
                    help='Result folder path [default: results]')
parser.add_argument('--test', type=bool, default=False, # only used in test server
                    help='Get performance on test data [default: False]')

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
if FLAGS.test:
    TEST_RESULT_DIR = os.path.join(RESULT_DIR, "test")
    if not os.path.exists(TEST_RESULT_DIR):
        os.makedirs(TEST_RESULT_DIR)
    LOG_FOUT = open(os.path.join(TEST_RESULT_DIR, 'log_evaluate.txt'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')
else:
    VAL_RESULT_DIR = os.path.join(RESULT_DIR, "test")
    if not os.path.exists(VAL_RESULT_DIR):
        os.makedirs(VAL_RESULT_DIR)
    LOG_FOUT = open(os.path.join(VAL_RESULT_DIR, 'log_evaluate.txt'), 'w')
    LOG_FOUT.write(str(FLAGS)+'\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
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

    eval_one_epoch(sess, ops, data_input)


def eval_one_epoch(sess, ops, data_input):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    loss_sum = 0

    num_batches = data_input.num_val // BATCH_SIZE
    acc_a_sum = [0] * 5
    acc_s_sum = [0] * 5

    preds = []
    labels_total = []
    acc_a = [0] * 5
    acc_s = [0] * 5
    for batch_idx in range(num_batches):
        if "io" in MODEL_FILE:
            imgs, labels = data_input.load_one_batch(BATCH_SIZE, "val")
            feed_dict = {ops['imgs_pl']: imgs,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training}
        else:
            imgs, others, labels = data_input.load_one_batch(BATCH_SIZE, "val")
            feed_dict = {ops['imgs_pl'][0]: imgs,
                         ops['imgs_pl'][1]: others,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training}

        loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                    feed_dict=feed_dict)

        preds.append(pred_val)
        labels_total.append(labels)
        loss_sum += np.mean(np.square(np.subtract(pred_val, labels)))
        for i in range(5):
            acc_a[i] = np.mean(np.abs(np.subtract(pred_val[:, 1], labels[:, 1])) < (1.0 * (i+1) / 180 * scipy.pi))
            acc_a_sum[i] += acc_a[i]
            acc_s[i] = np.mean(np.abs(np.subtract(pred_val[:, 0], labels[:, 0])) < (1.0 * (i+1) / 20))
            acc_s_sum[i] += acc_s[i]

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    for i in range(5):
        log_string('eval accuracy (angle-%d): %f' % (float(i+1), (acc_a_sum[i] / float(num_batches))))
        log_string('eval accuracy (speed-%d): %f' % (float(i+1), (acc_s_sum[i] / float(num_batches))))

    preds = np.vstack(preds)
    labels = np.vstack(labels_total)

    a_error, s_error = mean_max_error(preds, labels, dicts=get_dicts())
    log_string('eval error (mean-max): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = max_error(preds, labels)
    log_string('eval error (max): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = mean_topk_error(preds, labels, 5)
    log_string('eval error (mean-top5): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = mean_error(preds, labels)
    log_string('eval error (mean): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))

    print (preds.shape, labels.shape)
    np.savetxt(os.path.join(VAL_RESULT_DIR, "preds_val.txt"), preds)
    np.savetxt(os.path.join(VAL_RESULT_DIR, "labels_val.txt"), labels)
    # plot_acc(preds, labels)


def test():
    with tf.device('/gpu:'+str(GPU_INDEX)):
        if 'pn' in MODEL_FILE:
            data_input = provider.Provider2()
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

    test_one_epoch(sess, ops, data_input)


def test_one_epoch(sess, ops, data_input):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    loss_sum = 0

    num_batches = data_input.num_test // BATCH_SIZE
    acc_a_sum = [0] * 5
    acc_s_sum = [0] * 5

    preds = []
    labels_total = []
    acc_a = [0] * 5
    acc_s = [0] * 5
    for batch_idx in range(num_batches):
        if "io" in MODEL_FILE:
            imgs, labels = data_input.load_one_batch(BATCH_SIZE)
            feed_dict = {ops['imgs_pl']: imgs,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training}
        else:
            imgs, others, labels = data_input.load_one_batch(BATCH_SIZE)
            feed_dict = {ops['imgs_pl'][0]: imgs,
                         ops['imgs_pl'][1]: others,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training}

        loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                    feed_dict=feed_dict)

        preds.append(pred_val)
        labels_total.append(labels)
        loss_sum += np.mean(np.square(np.subtract(pred_val, labels)))
        for i in range(5):
            acc_a[i] = np.mean(np.abs(np.subtract(pred_val[:, 1], labels[:, 1])) < (1.0 * (i+1) / 180 * scipy.pi))
            acc_a_sum[i] += acc_a[i]
            acc_s[i] = np.mean(np.abs(np.subtract(pred_val[:, 0], labels[:, 0])) < (1.0 * (i+1) / 20))
            acc_s_sum[i] += acc_s[i]

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    for i in range(5):
        log_string('eval accuracy (angle-%d): %f' % (float(i+1), (acc_a_sum[i] / float(num_batches))))
        log_string('eval accuracy (speed-%d): %f' % (float(i+1), (acc_s_sum[i] / float(num_batches))))

    preds = np.vstack(preds)
    labels = np.vstack(labels_total)

    a_error, s_error = mean_max_error(preds, labels, dicts=get_dicts())
    log_string('eval error (mean-max): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = max_error(preds, labels)
    log_string('eval error (max): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = mean_topk_error(preds, labels, 5)
    log_string('eval error (mean-top5): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))
    a_error, s_error = mean_error(preds, labels)
    log_string('eval error (mean): angle:%.2f speed:%.2f' %
               (a_error / scipy.pi * 180, s_error * 20))

    print (preds.shape, labels.shape)
    np.savetxt(os.path.join(TEST_RESULT_DIR, "preds_val.txt"), preds)
    np.savetxt(os.path.join(TEST_RESULT_DIR, "labels_val.txt"), labels)
    # plot_acc(preds, labels)


def plot_acc(preds, labels, counts = 100):
    a_list = []
    s_list = []
    for i in range(counts):
        acc_a = np.abs(np.subtract(preds[:, 1], labels[:, 1])) < (20.0 / 180 * scipy.pi / counts * i)
        a_list.append(np.mean(acc_a))

    for i in range(counts):
        acc_s = np.abs(np.subtract(preds[:, 0], labels[:, 0])) < (15.0 / 20 / counts * i)
        s_list.append(np.mean(acc_s))

    print (len(a_list), len(s_list))
    a_xaxis = [20.0 / counts * i for i in range(counts)]
    s_xaxis = [15.0 / counts * i for i in range(counts)]

    auc_angle = np.trapz(np.array(a_list), x=a_xaxis) / 20.0
    auc_speed = np.trapz(np.array(s_list), x=s_xaxis) / 15.0

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(a_xaxis, np.array(a_list), label='Area Under Curve (AUC): %f' % auc_angle)
    plt.legend(loc='best')
    plt.xlabel("Threshold (angle)")
    plt.ylabel("Validation accuracy")
    plt.savefig(os.path.join(RESULT_DIR, "acc_angle.png"))
    plt.figure()
    plt.plot(s_xaxis, np.array(s_list), label='Area Under Curve (AUC): %f' % auc_speed)
    plt.xlabel("Threshold (speed)")
    plt.ylabel("Validation accuracy")
    plt.legend(loc='best')
    plt.savefig(os.path.join(RESULT_DIR, 'acc_spped.png'))

def plot_acc_from_txt(counts=100):
    preds = np.loadtxt(os.path.join(RESULT_DIR, "test/preds_val.txt"))
    labels = np.loadtxt(os.path.join(RESULT_DIR, "test/labels_val.txt"))
    print (preds.shape, labels.shape)
    plot_acc(preds, labels, counts)

def get_dicts(description="val"):
    if description == "train":
        raise NotImplementedError
    elif description == "val": # batch_size == 8
        return [120] * 4 + [111] + [120] * 4 + [109] + [120] * 9 + [89 - 87 % 8]
    elif description == "test": # batch_size == 8
        return [120] * 9 + [116] + [120] * 4 + [106] + [120] * 4 + [114 - 114 % 8]
    else:
        raise NotImplementedError

def mean_max_error(preds, labels, dicts):
    cnt = 0
    a_error = 0
    s_error = 0
    for i in dicts:
        print (preds.shape, cnt, cnt + i)
        a_error += np.max(np.abs(preds[cnt:cnt+i, 1] - labels[cnt:cnt+i, 1]))
        s_error += np.max(np.abs(preds[cnt:cnt+i, 0] - labels[cnt:cnt+i, 0]))
        cnt += i
    return a_error / float(len(dicts)), s_error / float(len(dicts))

def max_error(preds, labels):
    return np.max(np.abs(preds[:,1] - labels[:,1])), np.max(np.abs(preds[:, 0] - labels[:, 0]))

def mean_error(preds, labels):
    return np.mean(np.abs(preds[:,1] - labels[:,1])), np.mean(np.abs(preds[:,0] - labels[:,0]))

def mean_topk_error(preds, labels, k):
    a_error = np.abs(preds[:,1] - labels[:,1])
    s_error = np.abs(preds[:,0] - labels[:,0])
    return np.mean(np.sort(a_error)[::-1][0:k]), np.mean(np.sort(s_error)[::-1][0:k])

if __name__ == "__main__":
    if FLAGS.test: test()
    else: evaluate()
    # plot_acc_from_txt()
