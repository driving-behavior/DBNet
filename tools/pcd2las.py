""" 
Simple python scripts for 1) downsampling point clouds
2) converting point clouds from '.pcd' to '.las' format. 
Author: Jingkang Wang
Date: November 2017
Dependency: CloudCompare
"""

import argparse
import glob
import os
import time


def downsample(absolute_path):
    """
    Downsample point clouds (supported formats: las/pcd/...)
        :param absolute_path: directory of point clouds
    """
    files = glob.glob(absolute_path + "*.las")
    files.sort()
    files.sort(key=len)
    time_in = time.time()
    for f in files:
        os.system("CloudCompare.exe -SILENT \
                   -NO_TIMESTAMP -C_EXPORT_FMT LAS \
                   -O %s -SS RANDOM 16384" % f)
    print time.time() - time_in


def pcd2las(absolute_path):
    """
    Convert point clouds from las to pcd.
        :param absolute_path: directory of point clouds
    """
    print (absolute_path)
    files = glob.glob(absolute_path + "*.pcd")
    files.sort()
    files.sort(key=len)
    print (files)
    time_in = time.time()
    for f in files:
        os.system("CloudCompare.exe -SILENT \
                   -NO_TIMESTAMP -C_EXPORT_FMT LAS \
                   -O %s -SS RANDOM 16384" % f)
    print time.time() - time_in


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, required=True,
                        help='Input directory of point clouds')
    parser.add_argument('oper', type=str, default="downsample",
                        help='Operations to conduct')
    FLAGS = parser.parse_args()
    INPUT_DIR = FLAGS.input_dir
    OPER = FLAGS.oper

    assert (os.path.exists(INPUT_DIR))
    if (OPER == "downsample"):
        downsample(INPUT_DIR)
    else:
        pcd2las(INPUT_DIR)
