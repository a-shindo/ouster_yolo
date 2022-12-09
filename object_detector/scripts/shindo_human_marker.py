#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import cv2
import rospy
import tf
import torch
from pprint import pprint
from sensor_msgs.msg import Image
#from sensor_msgs.msg import CameraInfo
import message_filters
from cv_bridge import CvBridge
from importlib.metadata import metadata
from contextlib import closing
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2  
import sensor_msgs.point_cloud2 as pc2
import logging
import ros_numpy

# ROS preparation
rospy.init_node('human_tracker')

# yolov5 model import
model = torch.hub.load("/usr/local/lib/python3.8/dist-packages/yolov5", 'custom', path=os.environ['HOME']+'/catkin_ws/src/object_detector/config/best.pt',source='local')
#model = torch.hub.load("/usr/local/lib/python3.8/dist-packages/yolov5", 'custom', path=os.environ['HOME']+'/catkin_ws/src/object_detector/config/yolov5/yolov5s.pt',source='local')

# dpt history
dpt_history=[]

# csv
csv_path=os.environ['HOME']+"/catkin_ws/src/object_detector/monitor/results.csv"

# json
jsn_path=os.environ['HOME']+"/catkin_ws/src/object_detector/monitor/velocity.json"


try:
    os.remove(csv_path)
    os.remove(jsn_path)
except FileNotFoundError:
    pass

rect_list=[]

def pub_sub():
    global reflec_sub,point_sub
    # subscriber
    sub_pub_list=[]
    reflec_sub=message_filters.Subscriber("/ouster/reflec_image",Image)
    sub_pub_list.append(reflec_sub)
    range_sub=message_filters.Subscriber("/ouster/range_image",Image)
    sub_pub_list.append(range_sub)
    # publisher
    # marker_pub=message_filters.("/visualization_marker",Marker)
    # sub_pub_list.append(marker_pub)

    mf=message_filters.ApproximateTimeSynchronizer(sub_pub_list,10,0.5)
    
    # listener

    # broadcaster

    return mf

def img_position(obj_people):#,proj_mtx
    # リストの中に辞書が入っていて、その中に情報が埋め込まれてる
    """
    辞書の中身
    0. xmin_dpt
    1. ymin_dpt
    2. xmax_dpt
    3. ymax_dpt
    4. bd_center_x
    5. bd_center_y
    6. confidence
    """
    global  rect_list,one_person, bd_center_x,bd_center_y
    rect_list=[]
    for i,row in enumerate(obj_people.itertuples()):
        xmin_dpt=row.xmin
        ymin_dpt=row.ymin
        xmax_dpt=row.xmax
        ymax_dpt=row.ymax
        confidence=row.confidence
        bd_center_y=int((ymin_dpt+ymax_dpt)/2)
        bd_center_x=int((xmin_dpt+xmax_dpt)/2)
    
        # center_3d=dpt*(np.array([bd_center_x,bd_center_y,1]).T)
        #np.linalg.pinv(proj_mtx),
        #print(dpt)
        print("bd_center_x,bd_center_y,",[bd_center_x,bd_center_y,])
        #print(center_3d)#center_3d=dpt*np.dot(np.array([bd_center_x,bd_center_y,1]).T)#np.linalg.pinv(proj_mtx),
        one_person={
            'xmin_dpt':int(xmin_dpt),
            'ymin_dpt':int(ymin_dpt),
            'xmax_dpt':int(xmax_dpt),
            'ymax_dpt':int(ymax_dpt),
            'bd_center_x':bd_center_x,
            'bd_center_y':bd_center_y,
            'confidence':confidence,
            }
        rect_list.append(one_person)
        print("one_person",one_person)
    # return rect_list

def PcdCallback(point_cloud): 
    global  rect_list
    pc = ros_numpy.numpify(point_cloud)
    # point_cloud_list = pc2.read_points(point_cloud)
    # print("point_cloud_list", point_cloud_list)
    # print("len(point_cloud_list)", len(point_cloud_list))
    # print(" len(point_cloud_list)",len(point_cloud_list), 1024*128)
    #point_cloud_list_1 = np.array(point_cloud_list)
    
    
    # point_cloud_list_131072 = point_cloud_list_1.reshape([131072, 9])
    # point_cloud_list_1024_128 = point_cloud_list_1.reshape([1024, 128, 9])
    # point_cloud_list_128_1024 = point_cloud_list_1.reshape([128,1024, 9])
    # # point_cloud_list_9 = point_cloud_list_131072[131071, :]
    # # point_cloud_list_ = point_cloud_list_128_1024[:, :]
    # point_cloud_list_1 = point_cloud_list_131072[129, :]
    # point_cloud_list_2_1 = point_cloud_list_1024_128[1, 0,:]
    # # point_cloud_list_2_2 = point_cloud_list_1024_128[0, 129,:]
    # point_cloud_list_3_1 = point_cloud_list_128_1024[1, 0, :]
    # point_cloud_list_3_2 = point_cloud_list_128_1024[0, 129, :]
    # point_cloud_list_0_0 = point_cloud_list_128_1024[0, 0, :]
    # point_cloud_list_0_1023 = point_cloud_list_128_1024[0, 1023, :]
    # point_cloud_list_127_0 = point_cloud_list_128_1024[127, 0, :]
    # point_cloud_list_127_1023 = point_cloud_list_128_1024[127, 1023, :]

    # # marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
    
    marker1= Marker()

    marker1.header.frame_id = "os_sensor"
    marker1.header.stamp = rospy.Time.now()
    
    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker1.type = 2
    marker1.id = 0

    # Set the scale of the marker
    marker1.scale.x = 0.3
    marker1.scale.y = 0.3
    marker1.scale.z = 0.3

    # Set the color
    marker1.color.r = 0.0
    marker1.color.g = 1.0
    marker1.color.b = 0.0
    marker1.color.a = 1.0
    # # marker2.color.r = 1.0
    # # marker2.color.g = 0.0
    # # marker2.color.b = 0.0
    # # marker2.color.a = 1.0

    # # Set the pose of the marker
    # # marker1.pose.position.x = point_cloud_list_1024_128[400, 1, 0]
    # # marker1.pose.position.y = point_cloud_list_1024_128[400, 1, 1]
    # # marker1.pose.position.z = point_cloud_list_1024_128[400, 1, 2]
    print("rect_list", rect_list)
    if len(rect_list)>0:
        print(pc.shape)
        print([rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]])
        print(pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]])
        
        if not pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]][0] == 0 and not pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]][1] == 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]][2]
            # print("marker1.pose.position.x1", marker1.pose.position.x)
        elif not pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]][0] == 0 and not pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]][1]== 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]][2]
            # print("marker1.pose.position.x2", marker1.pose.position.x)
        elif not pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+1][0] == 0 and not pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+1][1] == 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+1][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+1][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+1][2]
            # print("marker1.pose.position.x3", marker1.pose.position.x)
        elif not pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+1][0] == 0 and not pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+1][1] == 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]][2]
            # print("marker1.pose.position.x4", marker1.pose.position.x)
        elif not pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]-1][0] == 0 and not pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]-1][1]-1 == 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]-1][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]-1][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]-1][2]
            # print("marker1.pose.position.x5", marker1.pose.position.x)
        elif not pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]+1][0] == 0 and not pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]+1][1] == 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]+1][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]+1][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]+1][2]
            # print("marker1.pose.position.x6", marker1.pose.position.x)
        elif not pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]+1][0] == 0 and not pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]+1][1]+1 == 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]+1][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]+1][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]+1][2]
        elif not pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]-1][0] == 0 and not pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]-1][1]-1 == 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]-1][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]-1][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"]+1, rect_list[0]["bd_center_x"]-1][2]
        elif not pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]-1][0] == 0 and not pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]-1][1]-1 == 0:
            marker1.pose.position.x = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]-1][0]
            marker1.pose.position.y = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]-1][1]
            marker1.pose.position.z = pc[rect_list[0]["bd_center_y"]-1, rect_list[0]["bd_center_x"]-1][2]
        # else:
        #     marker1.pose.position.x = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+2][0]
        #     marker1.pose.position.y = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+2][1]
        #     marker1.pose.position.z = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+2][2]
        print("marker1.pose.position.x", marker1.pose.position.x, marker1.pose.position.y)
        
    #     # while not rospy.is_shutdown():
        marker_pub.publish(marker1)
    # # rospy.rostime.wallsleep(1.0)    
    pass

def writeLog(rect_list,now):
    # if len(rect_list)>0:
    #     one_person=rect_list[0].tolist()
    #     one_person.insert(0,now)
    #     if len(dpt_history)>=2:
    #         # velocity_3d=np.sqrt((dpt_history[-1][1]-dpt_history[-2][1])**2+(dpt_history[-1][2]-dpt_history[-2][2])**2+(dpt_history[-1][3]-dpt_history[-2][3])**2)/(dpt_history[-1][0]-dpt_history[-2][0])
    #         velocity=(dpt_history[-1][3]-dpt_history[-2][3])/(dpt_history[-1][0]-dpt_history[-2][0])
            
    #         one_person.insert(len(one_person),velocity)
    #         if rect_list[0].tolist()[2]!=0 and velocity!=0:
    #             print(velocity)
    #             dpt_history.append(one_person)
    #     else:
    #         one_person.insert(len(one_person),0)
    #         dpt_history.append(one_person)
        
    #     np.savetxt(csv_path,dpt_history,delimiter=",")
    pass


    

def end_func(thre):
    data=np.loadtxt(csv_path,delimiter=",")
    z_list=data[:,3]
    vel_list=data[:,-1]
    vel_list=np.where(vel_list<-thre,0,vel_list)
    vel_list=np.where(vel_list>thre,0,vel_list)

    vel_info={
        "z_ave":np.average(z_list[-10:]),
        "z_latest":z_list[-1],
        "vel_z_ave":np.average(vel_list),
        "vel_z_md":np.median(vel_list),
        "vel_z_sd":np.std(vel_list),
    }
    #print(vel_list)
    #print(np.average(vel_list))
    jsn=open(jsn_path,"w")
    json.dump(vel_info,jsn)
    jsn.close()
    pass




def ImageCallback(box_data,dpt_data):#,info_data
    # try:
    # unpack arrays
    # print("ImageCallback")
    now=time.time()
    box_array = np.frombuffer(box_data.data, dtype=np.uint8).reshape(box_data.height, box_data.width, -1)
    box_array=np.nan_to_num(box_array)
    box_array= box_array[:, :, 0]
    #box_array=cv2.cvtColor(box_array,cv2.COLOR_GRAY2RGBA)
    dpt_array = np.frombuffer(dpt_data.data, dtype=np.uint8).reshape(dpt_data.height, dpt_data.width, -1)
    dpt_array=np.nan_to_num(dpt_array)
    #print("dpt_array",dpt_array)
    #proj_mtx=np.array(info_data.P).reshape(3,4)
    # object recognition
    results=model(box_array)
    # print("results", results)
    # print(bd_boxes)
    # results.render()
    # cv2.imshow("detected",results.imgs[0])

    objects=results.pandas().xyxy[0]
    obj_people=objects[objects['name']=='person']
    # rect_list=img_position(box_array,dpt_array,obj_people)#,proj_mtx
    img_position(obj_people)#,proj_mtx
    writeLog(rect_list,now)
    if len(dpt_history)>=100:
        reflec_sub.unregister()
        # dpt_sub.unregister()
        # info_sub.unregister()
        end_func(1500)
        rospy.on_shutdown(end_func)

    

    # except Exception:
            # exc_type, exc_obj, exc_tb = sys.exc_info()
            # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            # pprint(exc_type, fname, exc_tb.tb_lineno)


# topicName_reflec="/ouster/reflec_image"
# topicName_dpt="/ouster/range_image"
# topicName_point="/points"




mf=pub_sub()
# mf.registerCallback(ImageCallback_realsense)
mf.registerCallback(ImageCallback)

rospy.Subscriber("/ouster/points", PointCloud2, PcdCallback)
marker_pub=rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
rospy.spin()
