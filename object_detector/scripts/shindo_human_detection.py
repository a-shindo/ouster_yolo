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
import logging

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


def pub_sub():
    global reflec_sub,dpt_sub
    # subscriber
    sub_list=[]
    reflec_sub=message_filters.Subscriber(topicName_reflec,Image)
    sub_list.append(reflec_sub)
    dpt_sub=message_filters.Subscriber(topicName_dpt,Image)
    sub_list.append(dpt_sub)
    mf=message_filters.ApproximateTimeSynchronizer(sub_list,10,0.5)
    
    # publisher

    # listener

    # broadcaster

    return mf

def get_position(box_array,dpt_array,obj_people):#,proj_mtx
    # リストの中に辞書が入っていて、その中に情報が埋め込まれてる
    """
    辞書の中身
    0. xmin_dpt
    1. ymin_dpt
    2. xmax_dpt
    3. ymax_dpt
    4. bd_center_x
    5. bd_center_y
    6. center_3d
    7. confidence
    8. dpt

    """
    rect_list=[]

    for i,row in enumerate(obj_people.itertuples()):
        xmin_dpt=row.xmin
        ymin_dpt=row.ymin
        xmax_dpt=row.xmax
        ymax_dpt=row.ymax
        confidence=row.confidence
        bd_box=np.array(dpt_array[int(ymin_dpt):int(ymax_dpt),int(xmin_dpt):int(xmax_dpt)])
        dpt=np.nanmedian(bd_box) # 中央値
        bd_center_y=int((ymin_dpt+ymax_dpt)/2)
        bd_center_x=int((xmin_dpt+xmax_dpt)/2)
        poi=(bd_center_x, bd_center_y)
    
        center_3d=dpt*(np.array([bd_center_x,bd_center_y,1]).T)
        #np.linalg.pinv(proj_mtx),
        #print(dpt)
        #print([bd_center_x,bd_center_y,1])
        #print(center_3d)#center_3d=dpt*np.dot(np.array([bd_center_x,bd_center_y,1]).T)#np.linalg.pinv(proj_mtx),
        one_person={
            'xmin_dpt':int(xmin_dpt),
            'ymin_dpt':int(ymin_dpt),
            'xmax_dpt':int(xmax_dpt),
            'ymax_dpt':int(ymax_dpt),
            'bd_center_x':bd_center_x,
            'bd_center_y':bd_center_y,
            'center_3d':center_3d,
            'confidence':confidence,
            'dpt':dpt
            }
        rect_list.append(one_person)
        print(one_person)
    return rect_list

def writeLog(rect_list,now):
    if len(rect_list)>0:
        one_person=rect_list[0]['center_3d'].tolist()
        one_person.insert(0,now)
        if len(dpt_history)>=2:
            # velocity_3d=np.sqrt((dpt_history[-1][1]-dpt_history[-2][1])**2+(dpt_history[-1][2]-dpt_history[-2][2])**2+(dpt_history[-1][3]-dpt_history[-2][3])**2)/(dpt_history[-1][0]-dpt_history[-2][0])
            velocity=(dpt_history[-1][3]-dpt_history[-2][3])/(dpt_history[-1][0]-dpt_history[-2][0])
            
            one_person.insert(len(one_person),velocity)
            if rect_list[0]['center_3d'].tolist()[2]!=0 and velocity!=0:
                print(velocity)
                dpt_history.append(one_person)
        else:
            one_person.insert(len(one_person),0)
            dpt_history.append(one_person)
        
        np.savetxt(csv_path,dpt_history,delimiter=",")

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
    try:
        # unpack arrays
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

        # print(bd_boxes)
        #results.render()
        #cv2.imshow("detected",results.imgs[0])

        objects=results.pandas().xyxy[0]
        obj_people=objects[objects['name']=='person']
        rect_list=get_position(box_array,dpt_array,obj_people)#,proj_mtx
        writeLog(rect_list,now)
        if len(dpt_history)>=100:
            reflec_sub.unregister()
            dpt_sub.unregister()
            #info_sub.unregister()
            end_func(1500)
            rospy.on_shutdown(end_func)

    except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            pprint(exc_type, fname, exc_tb.tb_lineno)

# topicName_rgb="/camera3/camera/color/image_raw"
# topicName_dpt="/camera3/camera/aligned_depth_to_color/image_raw"
# topicName_camInfo="/camera3/camera/aligned_depth_to_color/camera_info"
topicName_reflec="/ouster/reflec_image"
topicName_dpt="/ouster/range_image"
#topicName_camInfo="/zed/zed_node/rgb/camera_info"



# subscribe
mf=pub_sub()
mf.registerCallback(ImageCallback)
rospy.spin()
