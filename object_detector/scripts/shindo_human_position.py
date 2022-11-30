#! /usr/bin/python3
# -*- coding: utf-8 -*-

#import os
#import sys
#import time
import csv
import numpy as np
import cv2
import rospy
import tf
import torch
#from pprint import pprint
from sensor_msgs.msg import Image
#from geometry_msgs.msg import PointStamped, Pose2D
#import message_filters
from cv_bridge import CvBridge
#import collections


csv_path = f"/home/ytpc2017d/catkin_ws/src/object_detector/csv/_2022-10-25-17-35-13.csv"

rows = []
with open(csv_path) as f:   
    reader = csv.reader(f)
    rows = [row for row in reader]

header = rows.pop(0)

data = np.float_(np.array(rows).T)

plt.xlabel(header[0])
plt.ylabel(header[2])


plt.plot(data[0], data[2], linestyle='solid', marker='o', color='red')
plt.plot(data[0], data[4], linestyle='solid', marker='o', color='blue')
plt.xlim(0, max(data[0]))
plt.ylim(0, 128)
# plt.savefig("_2022-10-25-17-35-13_y.png")
plt.show()

def human_position():

    


def get_position(self,rgb_array,dpt_array,obj_people,P):
    # alignでない場合のために、縮尺を導出
    y_rgb2dpt=dpt_array.shape[0]/rgb_array.shape[0]
    x_rgb2dpt=dpt_array.shape[1]/rgb_array.shape[1]
    now=rospy.get_time()-self.t_start
    # bounding boxをdpt画像に投射。
    rect_list=[]
    # 識別情報のデータベース
    # identifys = np.zeros((len(obj_people.itertuples()), 255))

    for i,row in enumerate(obj_people.itertuples()):
        xmin_dpt=row.xmin*x_rgb2dpt
        ymin_dpt=row.ymin*y_rgb2dpt
        xmax_dpt=row.xmax*x_rgb2dpt
        ymax_dpt=row.ymax*y_rgb2dpt
        confidence=row.confidence
        bd_box=np.array(dpt_array[int(ymin_dpt):int(ymax_dpt),int(xmin_dpt):int(xmax_dpt)])
        dpt=np.median(bd_box)
        bd_center_y=int((ymin_dpt+ymax_dpt)/2)
        bd_center_x=int((xmin_dpt+xmax_dpt)/2)
        center_3d=dpt*np.dot(np.linalg.pinv(P),np.array([bd_center_x,bd_center_y,1]).T)
        one_person=[now,int(xmin_dpt),int(ymin_dpt),int(xmax_dpt),int(ymax_dpt),bd_center_x,bd_center_y,center_3d,confidence,dpt]
        rect_list.append(one_person)

        # 個人識別とid付与
        # identifys[i] = personReidentification.infer(bd_box)
        # ids = Tracker.getIds(identifys, rect_list[:,0:4])
        # print(ids)
        

    return rect_list # [xmin,ymin,xmax,ymax,bd_center_x,bd_center_y,center_3d,confidence,dpt]


def callback(data):  # 呼び出し関数
    rospy.loginfo(data.data)  # pythonのprint
    
def listener():
    rospy.init_node('listener')  # ('Node名')
    rospy.Subscriber("/ouster/range_image", Image, callback)  # ("Topic名", 型, 関数)
    rospy.spin()

if __name__ == '__main__':
    listener()
