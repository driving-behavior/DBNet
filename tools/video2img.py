""" 
Simple python scripts for converting one video to continuous frames
Author: Jingkang Wang
Date: November 2017
Dependency: python-opencv
"""

import argparse
import math
import os
import sys

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Path of video')
parser.add_argument('-t', default=1.0, help='Time interval')
parser.add_argument('-o', default='./images', help='Dir of images')
FLAGS = parser.parse_args()

videoFile = FLAGS.i
imagesFolder = FLAGS.o
t_int = FLAGS.t

if videoFile == None:
    print ("[Error]: Please input path of video")
    sys.exit(0)

if not os.path.exists(videoFile):
    print ("[Error]: %s is not a valid video" % videoFile)
    sys.exit(0)

if not os.path.exists(imagesFolder): os.makedirs(imagesFolder)

cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate

count = 0
while(cap.isOpened()):
    frameId = cap.get(1)
    success, frame = cap.read()
    if not success:
        break
    #print frameId
    if (int(frameId) % math.floor(float(t_int) * frameRate) == 0):
        filename = imagesFolder + "/images_" + str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
        count += 1

    if (count % 100 == 0): print ("100 finished!")

cap.release()
print "Done!"
print ("FrameRate: %f" % frameRate)
print ("Total: %d" % count)
