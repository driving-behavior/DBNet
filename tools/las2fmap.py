""" 
Simple python scripts for extracting feature maps from point clouds
Author: Jingkang Wang
Date: November 2017
Dependency: python-opencv, numpy, pickle, scipy, laspy
"""

import argparse
import glob
import math
import os
import pickle

import numpy as np
from scipy.misc import imsave, imshow

import cv2
from laspy.base import Writer
from laspy.file import File


def lasReader(filename):
    """
    Read xyz points from single las file
        :param filename: path of single point cloud
    """
    f = File(filename, mode='r')
    x_max, x_min = np.max(f.x), np.min(f.x)
    y_max, y_min = np.max(f.y), np.min(f.y)
    z_max, z_min = np.max(f.z), np.min(f.z)
    return np.transpose(np.asarray([f.x, f.y, f.z])), \
            [(x_min, x_max), (y_min, y_max), (z_min, z_max)], f.header


def transform(merge, ranges, order=[0,1,2]):
    """
    Swap xyz axis
        :param merge: transformed xyz points
        :param ranges: specified shifts [default: None]
    """
    i = np.argsort(order)
    merge = merge[i,:]
    ranges = np.asarray(ranges)[i,:]
    return merge, ranges


def standardize(points, ranges=None):
    """
    Standardize points in point clouds
        :param filename: transformed xyz points
        :param ranges: specified shifts [default: None]
    """
    if ranges != None:
        points -= np.array([ranges[0][0], ranges[1][0], ranges[2][0]])
    else:
        x_min = np.min(points[:,0])
        y_min = np.min(points[:,1])
        z_min = np.min(points[:,2])
        points -= np.array(np.array([x_min, y_min, z_min]))
    return np.transpose(points), [(0, np.max(points[:,0])), \
            (0, np.max(points[:,1])), (0, np.max(points[:,2]))]


def rotate(img, angle=180):
    """
    Rotate images using opencv
        :param img: one image (opencv format)
        :param angle: rotated angle [default: 180]
    """
    rows, cols = img.shape[0], img.shape[1]
    rotation_matrix = cv2.getRotationMatrix2D((rows/2, cols/2), angle, 1)
    dst = cv2.warpAffine(img, rotation_matrix, (cols, rows))
    return dst


def rotate_about_center(src, angle, scale=1.):
    """
    Rotate images based on there centers
        :param src: one image (opencv format)
        :param angle: rotated angle
        :param scale: re-scaling images [default: 1.]
    """
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask opencv for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)


def feature_map(merge, ranges, alpha=0.2, beta=0.8, GSD=0.5):
    """
    Obatin feature maps from pound clouds
        :param merge: merged xyz points
        :param ranges: focused ranges
        :param alpha, beta, GSD: hyper-parameters in paper
    """
    (X_min, X_max) = ranges[0]
    (Y_min, Y_max) = ranges[1]
    (Z_min, Z_max) = ranges[2]

    W = int((X_max - X_min) / GSD) + 1
    H = int((Y_max - Y_min) / GSD) + 1
    print ("(W, H) = (" + str(W) + ", " + str(H) + ")")
    feature_map = np.zeros((W, H))

    net_dict = dict()
    for i in range(merge.shape[1]):
        if i % 1000000 == 0:
            print ("processed 1000000 points...")
        point = merge[:,i]
        x = int((point[0] - X_min) / GSD)
        y = int((point[1] - Y_min) / GSD)
        try:
            net_dict[(x, y)].append(i)
        except:
            net_dict[(x, y)] = [i]

    print ("mapping points...")
    # caculate the feature
    count = 0
    F_ij_min = 1000
    F_ij_max = -1000
    for i in range(W):
        for j in range(H):
            F_ij = 0

            try:
                h_min = 1000
                h_max = -1000
                for num in net_dict[(i, j)]:
                    point =  merge[:,num]
                    h_max = max(point[2], h_max)
                    h_min = min(point[2], h_min)

                Z_ijs = []
                W_ijs = []
                tol = 1e-5
                for num in net_dict[(i, j)]:
                    point =  merge[:,num]
                    Z_ij = point[2]
                    H_ij = Z_ij - Z_min
                    # obtain D_ij from Eqn.(5)
                    x_ij = (i + 0.5) * GSD + X_min
                    y_ij = (j + 0.5) * GSD + Y_min
                    D_ij = math.sqrt((point[0] - x_ij)**2 + (point[1] - y_ij)**2)
                    # obtain W_ij_XY and W_ij_J from Eqn.(4)
                    W_ij_XY = math.sqrt(2) * GSD / (D_ij + tol)
                    W_ij_H = H_ij * (h_min - Z_min) / (Z_max - h_max + tol)
                    # obtain W_ij from Eqn.(3)
                    W_ij = alpha * W_ij_XY + beta * W_ij_H
                    Z_ijs.append(Z_ij)
                    W_ijs.append(W_ij)

                for k in range(len(Z_ijs)):
                    # obtain feature value F_ij from Eqn.(2)
                    F_ij += W_ijs[k] * Z_ijs[k]

                F_ij /= sum(W_ijs)
                count += 1

                F_ij_min = min(F_ij, F_ij_min)
                F_ij_max = max(F_ij, F_ij_max)
            except: pass

            feature_map[i][j] = F_ij

    feature_map -= F_ij_min
    feature_map /= (F_ij_max - F_ij_min)
    feature_map *= 255

    return feature_map


def clean_map(fmap):
    """
    Clean the feature map
        :param fmap: feature map
    """
    # fmap = fmap[~(fmap==0).all(1)]
    fmap = fmap[(fmap != 0).sum(axis=1) >= 100, :]
    fmap = fmap[:, (fmap != 0).sum(axis=0) >= 50]

    return fmap


def resize(path, x_axis, y_axis):
    """
    Resize images
        :param path: path of an image
        :param x_axis: width of resized image
        :param y_axis: height of resized image
    """
    img = cv2.imread(path)
    new_image = cv2.resize(img, (x_axis, y_axis))
    cv2.imwrite(path, new_image)


def get_fmap(filename, dir1='gray', dir2='jet'):
    """
    Visualize feature maps
        :param dir1: path of gray images to be saved
        :param dir2: path of jet images to be saved
    """
    if not os.path.exists(dir1): os.mkdir(dir1)
    if not os.path.exists(dir2): os.mkdir(dir2)

    if not os.path.isfile(filename):
        print ("[Error]: \'%s\' is not a valid filename" % filename)
        return False

    merge, ranges, _ = lasReader(filename)
    merge, ranges = standardize(merge, ranges)
    print ("standardized point clouds")
    print ("total: " + str(merge.shape[1]) + " points")

    # transform x,y,z axis: 0,2,1
    merge, ranges = transform(merge, ranges, order=[1, 2, 0])

    # clean the feature map
    fmap = clean_map(feature_map(merge, ranges=ranges, GSD=0.05))
    cv2.imwrite(os.path.join(dir1, '%s.jpg' % filename[:-4]), \
                rotate_about_center(fmap, 180, 1.0))

    # uncomment the following line if you want to resize the feature map 
    # resize(os.path.join(dir1, '%s.jpg' % filename[:-4]), x_axis=1080, y_axis=270)

    gray = cv2.imread(os.path.join(dir1, '%s.jpg' % filename[:-4]))
    gray_single = gray[:,:,0]
    imC = cv2.applyColorMap(gray_single, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(dir1, '%s_jet_tmp.jpg' % filename[:-4]), imC)

    img = cv2.imread(os.path.join(dir1, '%s_jet_tmp.jpg' % filename[:-4]))
    cv2.imwrite(os.path.join(dir2, '%s_jet.jpg' % filename[:-4]), img)
    os.system("rm %s_jet_tmp.jpg" % os.path.join(dir1, filename[:-4]))

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default='',
                        help='Directory of las files [default: \'\']')
    parser.add_argument('-f', '--file', default='',
                        help='Specify one las file you want to convert [default: \'\']')
    FLAGS = parser.parse_args()

    d = FLAGS.dir
    f = FLAGS.file

    if f == '' and d == '':
        parser.print_help()
    elif f != '' and d != '':
        if not os.path.isdir(d):
            print ("[Error]: \'%s\' is not a valid directory!" % d)
        else:
            p = os.path.join(d, f)
            print (p)
            if get_fmap(p):
                print ("Finished!")
    elif f != '':
        p = f
        if get_fmap(p):
            print ("Finished!")
    else:
        if not os.path.isdir(d):
            print ("[Error]: \'%s\' is not a valid directory!" % d)
        else:
            files = sorted(glob.glob(os.path.join(d, "*.las")))
            count = 0
            for f in files:
                if get_fmap(f):
                    count += 1
                if count % 25 == 0 and count != 0:
                    print ("25 Finished!")


if __name__ == "__main__":
    main()
