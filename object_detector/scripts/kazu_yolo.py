#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import cv2
import numpy as np
import numpy as np
import cv2
import rospy
import sys
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from pprint import pprint
import message_filters

"""
他の端末の場合、諸々をローカルに入れる必要があるかもしれないので
yolov5にcdしたあと、
pip install -r requirements.txt
する。
"""

model=torch.hub.load('ultralytics/yolov5','yolov5s')
# model=torch.load("/catkin_ws/src/ytlab_whill_modules/scripts/yolov5/model_yolo_kazu.pt")

rotate_mode=False

def translate_rotation(after):
    #結局は転置
    before_x=after[1]
    before_y=after[0]
    return np.array([before_x,before_y])

def bd_box_info(rgb_data,dpt_data,obj):
    y_rgb2dpt=dpt_data.shape[0]/rgb_data.shape[0]
    x_rgb2dpt=dpt_data.shape[1]/rgb_data.shape[1]
    # y_rgb2dpt=1
    # x_rgb2dpt=1
    # print("rgb/dpt ratio : ",y_rgb2dpt,x_rgb2dpt)
    rect_data=[]
    for row in obj.itertuples():
        xmin_dpt=row.xmin*x_rgb2dpt
        ymin_dpt=row.ymin*y_rgb2dpt
        xmax_dpt=row.xmax*x_rgb2dpt
        ymax_dpt=row.ymax*y_rgb2dpt
        one_person=[int(xmin_dpt),int(ymin_dpt),int(xmax_dpt),int(ymax_dpt)]
        rect_data.append(one_person)
    return np.array(rect_data)

def draw_rect(img,bd_box,dpt_lists):
    for i,(person,dpt_list) in enumerate(zip(bd_box,dpt_lists)):
        cv2.rectangle(
            img,
            pt1=(person[0],person[1]),
            pt2=(person[2],person[3]),
            color=(255,255,255)
            )
        cv2.putText(
          img,
          text=f"P{i} : {str(int(dpt_list[0]))}",
          org=(person[0],person[1]),
          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
          fontScale=1.0,
          color=(125,125,125),
          thickness=2,
          )
    
    return img

def get_dpt(img,bd_boxes,P):
    dpt_list=[]
    for i, bd_box in enumerate(bd_boxes):
        human=np.array(img[bd_box[1]:bd_box[3],bd_box[0]:bd_box[2]])
        print(f"depth info of person {i}  average : ",np.average(human),"median : ",np.median(human))
        dpt=np.median(human)
        center=np.array([np.average([bd_box[0],bd_box[2]]),np.average([bd_box[1],bd_box[3]])])
        pt_3d=get_drc(center,P,dpt)
        dpt_list.append([dpt,pt_3d])
    return dpt_list

def get_drc(pt,P,dpt):
    print("center : ",pt)
    if rotate_mode:
      pt_original=translate_rotation(pt)
    else:
      pt_original=np.array((pt[0],pt[1]))
    pt_original=np.array((pt_original[0],pt_original[1],1)).T
    print("center after modified : ",pt_original)
    pt_3d=np.dot(np.linalg.pinv(P),pt_original)*dpt
    print("center 3d : ",pt_3d)
    pt_3d=np.array([pt_3d[1],pt_3d[0],pt_3d[2]])
    return pt_3d


def publish_point(dpt_list):
  for dpt_and_point in dpt_list:
    dpt_point=dpt_and_point[1]
    pub=rospy.Publisher("publisher_point",Point, queue_size=10)
    
    print("###### DEBUG ROI STARTS ######")
    pub.publish(dpt_point[0],dpt_point[1],dpt_point[2])
    print("###### DEBUG ROI ENDS ######")

def publish_PointStamped(dpt_list):
  for dpt_and_point in dpt_list:
    dpt_point=dpt_and_point[1]
    pub=rospy.Publisher("publisher_point",PointStamped, queue_size=30)
    point=PointStamped()
    point.header.stamp=rospy.Time.now()
    point.header.frame_id="/camera_link"
    # point.point.x=1.0
    # point.point.y=1.0
    # point.point.z=1.0
    point.point.x=dpt_point[2]/1000
    point.point.y=-dpt_point[1]/1000
    point.point.z=-dpt_point[0]/1000
    print("###### DEBUG ROI STARTS ######")
    pub.publish(point)
    print("###### DEBUG ROI ENDS ######")



def ImageCallback(rgb_data,dpt_data,info_data):
    try:
        rgb_array = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(rgb_data.height, rgb_data.width, -1)
        dpt_array = np.frombuffer(dpt_data.data, dtype=np.uint16).reshape(dpt_data.height, dpt_data.width, -1)
        Proj_mtx=np.array(info_data.P).reshape(3,4)
        # print(Proj_mtx)
        rgb_array=cv2.cvtColor(rgb_array,cv2.COLOR_BGR2RGB)
        print("size:",rgb_array.shape,dpt_array.shape)
        # bridge=CvBridge()
        # rgb_array = bridge.imgmsg_to_cv2(rgb_data)
        rgb_array=np.nan_to_num(rgb_array)
        dpt_array=np.nan_to_num(dpt_array)
        rgb_array=cv2.cvtColor(rgb_array,cv2.COLOR_BGR2RGB)
        if rotate_mode:
          rgb_array=cv2.rotate(rgb_array,cv2.ROTATE_90_COUNTERCLOCKWISE)
          dpt_array=cv2.rotate(dpt_array,cv2.ROTATE_90_COUNTERCLOCKWISE)
        # rospy.loginfo(rgb_array)
        dpt_array_show=(dpt_array-np.min(dpt_array))/(np.max(dpt_array)-np.min(dpt_array))
        results=model(rgb_array)
        objects=results.pandas().xyxy[0]
        obj_person=objects[objects['name']=='person']
        # print(obj_person)
        bd_boxes=bd_box_info(rgb_array,dpt_array,obj_person)
        
        dpt_list=get_dpt(dpt_array,bd_boxes,Proj_mtx)
        print("dpt_list",dpt_list)
        # publish_point(dpt_list)
        publish_PointStamped(dpt_list)
        dpt_array_show=draw_rect(dpt_array_show,bd_boxes,dpt_list)
        # print(bd_boxes)
        results.render()
        cv2.imshow("detected",results.imgs[0])
        cv2.imshow("depth",dpt_array_show)
        #cv2.imshow("depth",dpt_array)
        cv2.waitKey(1)
    except Exception as err:
        rospy.logerr(err)        

rospy.init_node('listener_camera')
rgb_sub=message_filters.Subscriber("/camera/color/image_raw",Image)
dpt_sub=message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",Image)
# dpt_sub=message_filters.Subscriber("/camera/depth/image_rect_raw",Image)
info_sub=message_filters.Subscriber("/camera/depth/camera_info",CameraInfo)
mf=message_filters.ApproximateTimeSynchronizer([rgb_sub,dpt_sub,info_sub],100,0.5)
mf.registerCallback(ImageCallback)
rospy.spin()

"""
rostopic echo /camera/depth/camera_info

header: 
  seq: 48
  stamp: 
    secs: 1657102672
    nsecs: 460047245
  frame_id: "camera_depth_optical_frame"
height: 480
width: 848
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [425.7173156738281, 0.0, 427.8653869628906, 0.0, 425.7173156738281, 244.0027618408203, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [425.7173156738281, 0.0, 427.8653869628906, 0.0, 0.0, 425.7173156738281, 244.0027618408203, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---




rostopic echo /camera/color/camera_info

header: 
  seq: 29
  stamp: 
    secs: 1657102806
    nsecs: 842561483
  frame_id: "camera_color_optical_frame"
height: 720
width: 1280
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [919.1873168945312, 0.0, 657.2086181640625, 0.0, 920.092041015625, 364.79913330078125, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [919.1873168945312, 0.0, 657.2086181640625, 0.0, 0.0, 920.092041015625, 364.79913330078125, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False


視野角の部分はかめら行列から何とかなるかもだけどぱっとわからないので試行錯誤
d435の場合
単位: deg
depth : 85.2 x 58   x 94
color : 69.4 x 42.5 x 77
"""