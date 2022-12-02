#! /usr/bin/env python
# -*- coding: utf-8 -*-
 
import csv
import numpy as np
import pandas as pd
import rospy
import cv2
import sys
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ouster import client
from importlib.metadata import metadata
from contextlib import closing

csv_path = f"/home/ytpc2017d/catkin_ws/src/object_detector/csv/_2022-10-25-17-35-13.csv"

rows = []
with open(csv_path) as f:   
    reader = csv.reader(f)
    rows = [row for row in reader]

data = np.float_(np.array(rows).T)
# リストのデータを、横軸用のデータ、縦軸用のデータに直すために、np.array(rows)でnumpy型に直した後に、「.T」で転置
# このままだとデータが文字列型になっているので「np.float_( )」で浮動小数点に直す

#バウンディングボックスの中心点の決定
box_center_x1= (data[1]+data[3])/2
box_center_x= np.floor(box_center_x1) #小数点以下切り捨て
box_center_y1= (data[2]+data[4])/2
box_center_y= np.floor(box_center_y1)
box_center_xy= pd.DataFrame(box_center_x, box_center_y)
pd.set_option('display.max_rows', 132) #列を省略なしで表示,後ろの数字は表示する行数
print(box_center_xy)


def callback(data):  # 呼び出し関数
    rospy.loginfo(data.data)  # pythonのprint
    
def listener():
    rospy.init_node('listener')  # ('Node名')
    rospy.Subscriber("/ouster/range_image", Image, callback)  # ("Topic名", 型, 関数)
    rospy.spin()

if __name__ == '__main__':
    listener()


#scan = LidarScan(h, w, info.format.udp_profile_lidar)
xyzlut = client.XYZLut(metadata) #call cartesian lookup table 
#xyz_destaggered = client.destagger(metadata, xyzlut(scan)) #to adjust for the pixel staggering that is inherent to Ouster lidar sensor raw data
#xyz = xyzlut(scan.field(client.ChanField.RANGE))
#range_roi = range_val[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] #xyxy is the X Y coordinates of the bounding box
#poi = np.unravel_index(range_roi.argmin(), range_roi.shape) #take the (x,y) coordinates of closest point within roi 

