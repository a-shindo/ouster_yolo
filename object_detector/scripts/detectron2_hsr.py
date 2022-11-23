from detectron2_core import *
import os
import sys
import time
import csv
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import rospy
import tf
import torch
from pprint import pprint
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped, Pose2D
import message_filters
from cv_bridge import CvBridge


"""
model type
OD: object detection
IS: instance segmentation
LVIS: LVinstance segmentation
PS: panoptic segmentation
KP: keypoint detection
"""
detector=Detector(model_type="KP")

topic_name = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'

detector.onROS(topicName=topic_name)

