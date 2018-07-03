from __future__ import print_function

import random
from math import ceil

import laspy
import numpy as np
import scipy.misc
import glob
import os


class Provider:
    def __init__(self, train_dir='data/dbnet-2018/train/',
                 val_dir='data/dbnet-2018/val/',
                 test_dir="data/dbnet-2018/test/"):

        self.__initialize__(train_dir, val_dir, test_dir)
        self.read()
        self.transform()

    def __initialize__(self, train_dir, val_dir, test_dir):
        self.X_train1, self.X_train2 = [], []
        self.Y_train1, self.Y_train2 = [], []
        self.X_val1, self.X_val2 = [], []
        self.Y_val1, self.Y_val2 = [], []
        self.X_test1, self.X_test2 = [], []
        self.x_train1, self.x_train2 = [], []
        self.x_val1, self.x_val2 = [], []
        self.x_test1, self.x_test2 = [], []
        self.train_pointer = 0
        self.val_pointer = 0
        self.test_pointer = 0
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.cache_train = False
        self.cache_val = False
        self.cache_test = False

    def read(self, filename="behavior.csv"):
        """
        Read data and labels
            :param filename: filename of labels
        """
        train_sub = glob.glob(os.path.join(self.train_dir, "*"))
        val_sub = glob.glob(os.path.join(self.val_dir, "*"))
        test_sub = glob.glob(os.path.join(self.test_dir, "*"))
        train_sub.sort()
        val_sub.sort()
        self.read_from(train_sub, filename, "train")
        self.read_from(val_sub, filename, "val")
        self.read_from(test_sub, filename, "test")

    def read_from(self, sub_folders, filename, description="train"):
        if description == "train":
            X_train1, X_train2 = self.X_train1, self.X_train2
            Y_train1, Y_train2 = self.Y_train1, self.Y_train2
        elif description == "val":
            X_train1, X_train2 = self.X_val1, self.X_val2
            Y_train1, Y_train2 = self.Y_val1, self.Y_val2
        elif description == "test":
            X_train1, X_train2 = self.X_test1, self.X_test2
        else:
            raise NotImplementedError

        if not description == "test":
            for folder in sub_folders:
                with open(os.path.join(folder, filename)) as f:
                    count = 0
                    for line in f:
                        line = line.rstrip()
                        X_train1.append(os.path.join(folder, "dvr_66x200/" + str(count) + ".jpg"))
                        X_train2.append(os.path.join(folder, "points_16384/" + str(count) + ".las"))
                        Y_train1.append((float(line.split(",")[1]) - 20) / 20)
                        Y_train2.append(float(line.split(",")[0]) * scipy.pi / 180)
                        count += 1
        else:
            for folder in sub_folders:
                for count in range(len(glob.glob(os.path.join(folder, "dvr_66x200/*.jpg")))):
                    X_train1.append(os.path.join(folder, "dvr_66x200/" + str(count) + ".jpg"))
                    X_train2.append(os.path.join(folder, "points_16384/" + str(count) + ".las"))

        if not description == "test":
            c = list(zip(X_train1, X_train2, Y_train1, Y_train2))
        else:
            c = list(zip(X_train1, X_train2))
        random.shuffle(c)

        if description == "train":
            self.X_train1, self.X_train2, self.Y_train1, self.Y_train2 = zip(*c)
        elif description == "val":
            self.X_val1, self.X_val2, self.Y_val1, self.Y_val2 = zip(*c)
        elif description == "test":
            self.X_test1, self.X_test2 = zip(*c)
        else:
            raise NotImplementedError

    def transform(self):
        self.X_train1, self.X_train2 = np.asarray(self.X_train1), np.asarray(self.X_train2)
        self.Y_train = np.transpose(np.asarray((self.Y_train1, self.Y_train2)))

        self.X_val1, self.X_val2 = np.asarray(self.X_val1), np.asarray(self.X_val2)
        self.Y_val = np.transpose(np.asarray((self.Y_val1, self.Y_val2)))

        self.num_test = len(self.X_test1)
        self.X_test1, self.X_test2 = np.asarray(self.X_test1), np.asarray(self.X_test2)

        self.num_train = len(self.Y_train)
        self.num_val = len(self.Y_val)

    def load_one_batch(self, batch_size, description='train', shape=[66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        if description == 'train':
            if not self.cache_train:
                print ("Loading training data ...")
                for i in range(self.num_train):
                    self.x_train1.append(scipy.misc.imresize(scipy.misc.imread(
                              self.X_train1[i]), shape) / 255.0)
                    infile = laspy.file.File(self.X_train2[i])
                    data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                    self.x_train2.append(data)
                    self.cache_train = True
                print ("Finished loading!")
            
            for i in range(0, batch_size):
                index = (self.train_pointer + i) % len(self.X_train1)
                x_out1.append(self.x_train1[index])
                x_out2.append(self.x_train2[index])
                y_out.append(self.Y_train[index])
                self.train_pointer += batch_size

        elif description == "val":
            if not self.cache_val:
                print ("Loading validation data ...")
                for i in range(0, self.num_val):
                    self.x_val1.append(scipy.misc.imresize(scipy.misc.imread(
                                self.X_val1[i]), shape) / 255.0)
                    infile = laspy.file.File(self.X_val2[i])
                    data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                    self.x_val2.append(data)
                    self.cache_val = True
                print ("Finished loading!")
                    
            for i in range(0, batch_size):
                index = (self.val_pointer + i) % len(self.X_val1)
                x_out1.append(self.x_val1[index])
                x_out2.append(self.x_val2[index])
                y_out.append(self.Y_val[index])
                self.val_pointer += batch_size

        elif description == "test":
            if not self.cache_test:
                print ("Loading testing data ...")
                for i in range(0, self.num_test):
                    self.x_test1.append(scipy.misc.imresize(scipy.misc.imread(
                                self.X_test1[i]), shape) / 255.0)
                    infile = laspy.file.File(self.X_test2[i])
                    data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                    self.x_test2.append(data)
                    self.cache_test = True
                print ("Finished loading!")

            for i in range(0, batch_size):
                index = (self.test_pointer + i) % len(self.X_test1)
                x_out1.append(self.x_test1[index])
                x_out2.append(self.x_test2[index])
                self.test_pointer += batch_size

        else:
            raise NotImplementedError

        if not description == "test":
            return np.stack(x_out1), np.stack(x_out2), np.stack(y_out)
        else:
            return np.stack(x_out1), np.stack(x_out2) 

    def load_val_all(self, batch_size, shape=[66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        index = 0
        iteration = int(ceil(len(self.X_val1) / float(batch_size)))
        for i in range(iteration):
            xs1 = []
            xs2 = []
            ys = []
            if i == iteration - 1:
                for i in range(index, len(self.X_val1)):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                               self.X_val1[i]), shape) / 255.0)
                    infile = laspy.file.File(self.X_val2[i])
                    data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                    xs2.append(data)

                    ys.append(self.Y_val[i])
            else:
                for i in range(0, batch_size):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                               self.X_val1[index + i]), shape) / 255.0)
                    infile = laspy.file.File(self.X_val2[index + i])
                    data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                    xs2.append(data)
                    ys.append(self.Y_val[index + i])

                index += batch_size

            x_out1.append(np.stack(xs1))
            x_out2.append(np.stack(xs2))
            y_out.append(np.stack(ys))
        return np.asarray(x_out1), np.asarray(x_out2), np.asarray(y_out)


class DVR_Provider:
    def __init__(self, input_dir='data/demo/DVR/'):
        self.xs = []
        self.ys1 = []   # vehicle speeds
        self.ys2 = []   # wheel angles
        self.ys = []
        self.train_pointer = 0
        self.val_pointer = 0
        self.path = input_dir
        self.read()
        self.num_images = len(self.xs)
        self.transform()
        self.split_set(0.8)
        self.suffle()
        print (self.xs)

    def read(self, filename="data.csv"):
        """
        Read data and labels
            :param filename: path of labels
        """
        with open(self.path[0:-4] + filename) as f:
            count = 0
            for line in f:
                line = line.rstrip()
                self.xs.append(self.path + str(count) + ".jpg")
                self.ys1.append((float(line.split(",")[1]) - 20) / 20)
                self.ys2.append(float(line.split(",")[0]) * scipy.pi / 180)
                count += 1
        c = list(zip(self.xs, self.ys1, self.ys2))
        random.shuffle(c)
        self.xs, self.ys1, self.ys2 = zip(*c)

    def transform(self):
        c = list(zip(self.xs, self.ys1, self.ys2))
        xs, ys1, ys2 = zip(*c)
        self.xs = np.asarray(xs)
        self.ys = np.transpose(np.asarray((ys1, ys2)))

    def suffle(self):
        """
        Suffle the data and labels
        """
        c = list(zip(self.train_xs, self.train_ys))
        random.shuffle(c)
        self.train_xs, self.train_ys = zip(*c)

        c = list(zip(self.val_xs, self.val_ys))
        random.shuffle(c)
        self.val_xs, self.val_ys = zip(*c)

    def split_set(self, rate=0.8):
        """
        Split data into training and validation sets
            :param rate: split ratio (80-20 train-val by default)
        """
        self.train_xs = self.xs[: int(rate * self.num_images)]
        self.train_ys = self.ys[: int(rate * self.num_images)]
        self.val_xs = self.xs[int((rate) * self.num_images):]
        self.val_ys = self.ys[int((rate) * self.num_images):]
        self.num_train = len(self.train_ys)
        self.num_val = len(self.val_ys)

    def load_one_batch(self, batch_size, description='train', shape=[66, 200]):
        """
        Load one-batch data and labels
            :param batch_size
            :param description: load training or valiation data
            :param shape: resized image shape ([66, 200] by default)
        """
        x_out = []
        y_out = []
        if description == 'train':
            for i in range(0, batch_size):
                index = (self.train_pointer + i) % len(self.train_xs)
                x_out.append(scipy.misc.imresize(scipy.misc.imread(
                             self.train_xs[index]), shape) / 255.0)
                y_out.append(self.train_ys[index])
                self.train_pointer += batch_size
        else:
            for i in range(0, batch_size):
                index = (self.val_pointer + i) % len(self.val_xs)
                x_out.append(scipy.misc.imresize(scipy.misc.imread(
                             self.val_xs[index]), shape) / 255.0)
                y_out.append(self.val_ys[index])
                self.val_pointer += batch_size
        return np.stack(x_out), np.stack(y_out)

    def load_val_all(self, batch_size, shape=[66, 200]):
        """
        Load all validation data and labels
            :param batch_size
            :param shape: resized image shape ([66, 200] by default)
        """
        x_out = []
        y_out = []
        index = 0
        iteration = int(ceil(len(self.val_xs) / float(batch_size)))

        for i in range(iteration):
            xs = []
            ys = []
            if i == iteration - 1:
                for i in range(index, len(self.val_xs)):
                    xs.append(scipy.misc.imresize(scipy.misc.imread(
                              self.val_xs[i]), shape) / 255.0)
                    ys.append(self.val_ys[i])
            else:
                for i in range(0, batch_size):
                    xs.append(scipy.misc.imresize(scipy.misc.imread(
                              self.val_xs[index + i]), shape) / 255.0)
                    ys.append(self.val_ys[index + i])
                index += batch_size

            x_out.append(np.stack(xs))
            y_out.append(np.stack(ys))
        return np.asarray(x_out), np.asarray(y_out)


class DVR_FMAP_Provider:
    def __init__(self, input_dir1='data/demo/DVR/', 
                 input_dir2='data/demo/fmap/'):
        self.xs1 = []
        self.xs2 = []
        self.ys1 = []  # vehicle speeds
        self.ys2 = []  # wheel angles
        self.ys = []
        self.train_pointer = 0
        self.val_pointer = 0
        self.input_dir1 = input_dir1
        self.input_dir2 = input_dir2
        self.read()
        self.num_images = len(self.xs1)
        self.transform()
        self.split_set(0.8)
        self.suffle()

    def read(self, filename="data.csv"):
        with open(self.input_dir1[0:-4] + filename) as f:
            count = 0
            for line in f:
                line = line.rstrip()
                self.xs1.append(self.input_dir1 + str(count) + ".jpg")
                self.xs2.append(self.input_dir2 + str(count) + ".jpg")

                self.ys1.append((float(line.split(",")[1]) - 20) / 20)
                self.ys2.append(float(line.split(",")[0]) * scipy.pi / 180)
                count += 1
        c = list(zip(self.xs1, self.xs2, self.ys1, self.ys2))
        random.shuffle(c)
        self.xs1, self.xs2, self.ys1, self.ys2 = zip(*c)

    def transform(self):
        c = list(zip(self.xs1, self.xs2, self.ys1, self.ys2))
        xs1, xs2, ys1, ys2 = zip(*c)
        self.xs1, self.xs2 = np.asarray(xs1), np.asarray(xs2)
        self.ys = np.transpose(np.asarray((ys1, ys2)))

    def suffle(self):
        c = list(zip(self.train_xs1, self.train_xs2, self.train_ys))
        random.shuffle(c)
        self.train_xs1, self.train_xs2, self.train_ys = zip(*c)

        c = list(zip(self.val_xs1, self.val_xs2, self.val_ys))
        random.shuffle(c)
        self.val_xs1, self.val_xs2, self.val_ys = zip(*c)

    def split_set(self, rate=0.8):
        self.train_xs1 = self.xs1[: int(rate * self.num_images)]
        self.train_xs2 = self.xs2[: int(rate * self.num_images)]
        self.train_ys = self.ys[: int(rate * self.num_images)]
        self.val_xs1 = self.xs1[-int((1 - rate) * self.num_images):]
        self.val_xs2 = self.xs2[-int((1 - rate) * self.num_images):]
        self.val_ys = self.ys[-int((1 - rate) * self.num_images):]
        self.num_train = len(self.train_ys)
        self.num_val = len(self.val_ys)

    def load_one_batch(self, batch_size, description='train', shape1=[66, 200], shape2=[66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        if description == 'train':
            for i in range(0, batch_size):
                index = (self.train_pointer + i) % len(self.train_xs1)
                x_out1.append(scipy.misc.imresize(scipy.misc.imread(
                              self.train_xs1[index]), shape1) / 255.0)
                x_out2.append(scipy.misc.imresize(scipy.misc.imread(
                              self.train_xs2[index]), shape2) / 255.0)
                y_out.append(self.train_ys[index])
                self.train_pointer += batch_size
        else:
            for i in range(0, batch_size):
                index = (self.val_pointer + i) % len(self.val_xs1)
                x_out1.append(scipy.misc.imresize(scipy.misc.imread(
                              self.val_xs1[index]), shape1) / 255.0)
                x_out2.append(scipy.misc.imresize(scipy.misc.imread(
                              self.val_xs2[index]), shape2) / 255.0)
                y_out.append(self.val_ys[index])
                self.val_pointer += batch_size

        return np.stack(x_out1), np.stack(x_out2), np.stack(y_out)

    def load_val_all(self, batch_size, shape1=[66, 200], shape2=[66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        index = 0
        iteration = int(ceil(len(self.val_xs1) / float(batch_size)))
        for i in range(iteration):
            xs1 = []
            xs2 = []
            ys = []
            if i == iteration - 1:
                for i in range(index, len(self.val_xs1)):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                               self.val_xs1[i]), shape1) / 255.0)
                    xs2.append(scipy.misc.imresize(scipy.misc.imread(
                               self.val_xs2[i]), shape2) / 255.0)
                    ys.append(self.val_ys[i])
            else:
                for i in range(0, batch_size):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                               self.val_xs1[index + i]), shape1) / 255.0)
                    xs2.append(scipy.misc.imresize(scipy.misc.imread(
                               self.val_xs2[index + i]), shape2) / 255.0)
                    ys.append(self.val_ys[index + i])
                index += batch_size

            x_out1.append(np.stack(xs1))
            x_out2.append(np.stack(xs2))
            y_out.append(np.stack(ys))
        return np.asarray(x_out1), np.asarray(x_out2), np.asarray(y_out)


class DVR_Points_Provider:
    def __init__(self, input_dir1='data/demo/DVR/', 
                 input_dir2='data/demo/points_16384/'):

        self.xs1 = []  # DVR
        self.xs2 = []  # points
        self.ys1 = []  # vehicle speeds
        self.ys2 = []  # wheel angles
        self.ys = []
        self.train_pointer = 0
        self.val_pointer = 0
        self.input_dir1 = input_dir1
        self.input_dir2 = input_dir2
        self.read()
        self.num_images = len(self.xs1)
        self.transform()
        self.split_set(0.8)
        self.suffle()

    def read(self, filename="data.csv"):
        with open(self.input_dir1[0:-4] + filename) as f:
            count = 0
            for line in f:
                line = line.rstrip()
                self.xs1.append(self.input_dir1 + str(count) + ".jpg")
                self.xs2.append(self.input_dir2 + str(count) + ".las")
                self.ys1.append((float(line.split(",")[1]) - 20) / 20)
                self.ys2.append(float(line.split(",")[0]) * scipy.pi / 180)
                count += 1
        c = list(zip(self.xs1, self.xs2, self.ys1, self.ys2))
        random.shuffle(c)
        self.xs1, self.xs2, self.ys1, self.ys2 = zip(*c)

    def transform(self):
        c = list(zip(self.xs1, self.xs2, self.ys1, self.ys2))
        xs1, xs2, ys1, ys2 = zip(*c)
        self.xs1, self.xs2 = np.asarray(xs1), np.asarray(xs2)
        self.ys = np.transpose(np.asarray((ys1, ys2)))

    def suffle(self):
        c = list(zip(self.train_xs1, self.train_xs2, self.train_ys))
        random.shuffle(c)
        self.train_xs1, self.train_xs2, self.train_ys = zip(*c)

        c = list(zip(self.val_xs1, self.val_xs2, self.val_ys))
        random.shuffle(c)
        self.val_xs1, self.val_xs2, self.val_ys = zip(*c)

    def split_set(self, rate=0.8):
        self.train_xs1 = self.xs1[: int(rate * self.num_images)]
        self.train_xs2 = self.xs2[: int(rate * self.num_images)]
        self.train_ys = self.ys[: int(rate * self.num_images)]
        self.val_xs1 = self.xs1[-int((1 - rate) * self.num_images):]
        self.val_xs2 = self.xs2[-int((1 - rate) * self.num_images):]
        self.val_ys = self.ys[-int((1 - rate) * self.num_images):]
        self.num_train = len(self.train_ys)
        self.num_val = len(self.val_ys)

    def load_one_batch(self, batch_size, description='train', shape=[66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        if description == 'train':
            for i in range(0, batch_size):
                index = (self.train_pointer + i) % len(self.train_xs1)
                x_out1.append(scipy.misc.imresize(scipy.misc.imread(
                              self.train_xs1[index]), shape) / 255.0)
                infile = laspy.file.File(self.train_xs2[index])
                data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                x_out2.append(data)
                y_out.append(self.train_ys[index])
                self.train_pointer += batch_size
        else:
            for i in range(0, batch_size):
                index = (self.val_pointer + i) % len(self.val_xs1)
                x_out1.append(scipy.misc.imresize(scipy.misc.imread(
                              self.val_xs1[index]), shape) / 255.0)
                infile = laspy.file.File(self.val_xs2[index])
                data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                x_out2.append(data)
                y_out.append(self.val_ys[index])
                self.val_pointer += batch_size
        return np.stack(x_out1), np.stack(x_out2), np.stack(y_out)

    def load_val_all(self, batch_size, shape=[66, 200]):
        x_out1 = []
        x_out2 = []
        y_out = []
        index = 0
        iteration = int(ceil(len(self.val_xs1) / float(batch_size)))
        for i in range(iteration):
            xs1 = []
            xs2 = []
            ys = []
            if i == iteration - 1:
                for i in range(index, len(self.val_xs1)):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                               self.val_xs1[i]), shape) / 255.0)
                    infile = laspy.file.File(self.val_xs2[i])
                    data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                    xs2.append(data)

                    ys.append(self.val_ys[i])
            else:
                for i in range(0, batch_size):
                    xs1.append(scipy.misc.imresize(scipy.misc.imread(
                               self.val_xs1[index + i]), shape) / 255.0)
                    infile = laspy.file.File(self.val_xs2[index + i])
                    data = np.vstack([infile.X, infile.Y, infile.Z]).transpose()
                    xs2.append(data)
                    ys.append(self.val_ys[index + i])

                index += batch_size

            x_out1.append(np.stack(xs1))
            x_out2.append(np.stack(xs2))
            y_out.append(np.stack(ys))
        return np.asarray(x_out1), np.asarray(x_out2), np.asarray(y_out)


def testProvider():
    instance0 = Provider()
    print (len(instance0.X_train1), len(instance0.X_train2), len(instance0.Y_train))
    print (len(instance0.X_val1), len(instance0.X_val2), len(instance0.Y_val))
    print (len(instance0.X_test1), len(instance0.X_test2))
    print (instance0.X_val1[:3], instance0.X_val2[:3])
    print (instance0.X_val2[0])
    x1_, x2_, y_ = instance0.load_val_all(16)
    print (len(x1_), len(x2_), len(y_))
    print (x2_[0].shape)
    for i in range(1):
        x1_, x2_, y_ = instance0.load_one_batch(156, 'train')
        print (x1_.shape, x2_.shape, y_.shape)
        print (np.asarray(instance0.x_train1).shape)
        x1_, x2_, y_ = instance0.load_one_batch(156, 'train')
        print (x1_.shape, x2_.shape, y_.shape)
        x1_, x2_, y_ = instance0.load_one_batch(156, 'val')
        print (x1_.shape, x2_.shape, y_.shape)
        print (np.asarray(instance0.x_val1).shape)
        x1_, x2_, y_ = instance0.load_one_batch(156, 'val')
        x1_, x2_ = instance0.load_one_batch(156, 'test')
        print (x1_.shape, x2_.shape)
        print (np.asarray(instance0.x_test1).shape)
        x1_, x2_ = instance0.load_one_batch(156, 'test')
    '''
    instance1 = DVR_Provider()
    print (len(instance1.xs), len(instance1.ys))
    print (len(instance1.train_xs), len(instance1.train_ys))
    print (len(instance1.val_xs), len(instance1.val_ys))
    print (instance1.val_xs[:3], instance1.val_ys[:3])
    x_, y_ = instance1.load_one_batch(16, 'train')
    print (x_.shape, y_.shape)
    x_, y_ = instance1.load_one_batch(16, 'val')
    print (x_.shape, y_.shape)
    x_, y_ = instance1.load_val_all(16)
    print (len(x_), len(y_))
    print (x_[0].shape)
    print (x_[len(x_) - 1].shape)

    instance2 = DVR_FMAP_Provider(input_dir1='data/demo/DVR/', input_dir2='data/demo/fmap/')
    print (len(instance2.xs1), len(instance2.xs2), len(instance2.ys))
    print (len(instance2.train_xs1), len(instance2.train_xs2), len(instance2.train_ys))
    print (len(instance2.val_xs1), len(instance2.val_xs2), len(instance2.val_ys))
    print (instance2.val_xs1[:3], instance2.val_xs2[:3], instance2.val_ys[:3])
    for i in range(1):
        x1_, x2_, y_ = instance2.load_one_batch(156, 'train')
        print (x1_.shape, x2_.shape, y_.shape)
    x1_, x2_, y_ = instance2.load_val_all(16)
    print (len(x1_), len(x2_), len(y_))
    print (x1_[0].shape)
    print (x2_[len(x1_) - 1].shape)
    print (y_[2].shape)

    instance3 = DVR_Points_Provider()
    print (len(instance3.xs1), len(instance3.xs2), len(instance3.ys))
    print (len(instance3.train_xs1), len(instance3.train_xs2), len(instance3.train_ys))
    print (len(instance3.val_xs1), len(instance3.val_xs2), len(instance3.val_ys))
    print (instance3.val_xs1[:3], instance3.val_xs2[:3], instance3.val_ys[:3])
    print (instance3.xs2[0])
    x1_, x2_, y_ = instance3.load_val_all(16)
    print (len(x1_), len(x2_), len(y_))
    print (x2_[0].shape)
    for i in range(1):
        x1_, x2_, y_ = instance3.load_one_batch(156, 'train')
        print (x1_.shape, x2_.shape, y_.shape)
    '''

if __name__ == "__main__":
    testProvider()
