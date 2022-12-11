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
from cv_bridge import CvBridge, CvBridgeError
from importlib.metadata import metadata
from contextlib import closing
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2  
import sensor_msgs.point_cloud2 as pc2
import logging
import ros_numpy
import matplotlib.pyplot as plt
from importlib.metadata import metadata
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import torch
from importlib.metadata import metadata
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo



cfg=get_cfg()
# Load model config and pretrained model
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5
cfg.MODEL_DEVICE="cuda"
predictor=DefaultPredictor(cfg)
bridge = CvBridge()

# ROS preparation
rospy.init_node('human_tracker')



# dpt history
dpt_history=[]

# csv
# csv_path=os.environ['HOME']+"/catkin_ws/src/object_detector/monitor/results.csv"

# json
# jsn_path=os.environ['HOME']+"/catkin_ws/src/object_detector/monitor/velocity.json"


# try:
#     os.remove(csv_path)
#     os.remove(jsn_path)
# except FileNotFoundError:
#     pass


def pub_sub():
    global reflec_sub,point_sub
    # subscriber
    sub_pub_list=[]
    reflec_sub=message_filters.Subscriber("/ouster/signal_image",Image)
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


def ImageCallback(box_data,dpt_data):#,info_data
    global rect_list
    now=time.time()
    cv_image = bridge.imgmsg_to_cv2(box_data, desired_encoding='bgr8')
    # print("cv_image",type(cv_image))
    signal_array = np.frombuffer(box_data.data,dtype=np.uint8).reshape(box_data.height, box_data.width, -1)
    signal_array=np.nan_to_num(signal_array)
 
    dpt_array = np.frombuffer(dpt_data.data, dtype=np.uint8).reshape(dpt_data.height, dpt_data.width, -1)
    dpt_array=np.nan_to_num(dpt_array)
   
    obj_people=predictor(cv_image)
    viz=Visualizer(cv_image[:,:,::-1],
    metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
    instance_mode=ColorMode.IMAGE_BW)
    # print("obj_people",obj_people)
    output=viz.draw_instance_predictions(obj_people["instances"].to("cpu"))
    cv2.imshow("Result",output.get_image()[:,:,::-1])
    key=cv2.waitKey(1) & 0xFF



    rect_list=img_position(obj_people)
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


def img_position(obj_people):
    global one_person, bd_center_px, bd_center_py, left_pt, right_pt,pxmin, rect_list
    rect_list=[]
    bounding_boxes = obj_people["instances"]._fields["pred_boxes"].tensor.cpu().numpy()
    for i in range(len(bounding_boxes)):
        # 左上座標
        left_pt = list(bounding_boxes[i][0:2])
        # 右下座標
        right_pt = list(bounding_boxes[i][2:4])
        pxmin=left_pt[1]
        pymin=left_pt[0]
        pxmax=right_pt[1]
        pymax=right_pt[0]
        bd_center_py=int((left_pt[1]+right_pt[1])/2)
        bd_center_px=int((left_pt[0]+right_pt[0])/2)
        print("bd_center_px,bd_center_py,",[bd_center_px,bd_center_py,])
        # リストの中に辞書が入っていて、その中に情報が埋め込まれてる
        """
        辞書の中身
        0. pxmin
        1. pymin
        2. pxmax
        3. pymax
        4. bd_center_px
        5. bd_center_py
        """
        one_person={
            'pxmin':left_pt[0],
            'pymin':left_pt[1],
            'pxmax':right_pt[0],
            'pymax':right_pt[1],
            'bd_center_px':bd_center_px,
            'bd_center_py':bd_center_py,
            }
        rect_list.append(one_person)
        print("one_person",one_person)
    return rect_list

def PcdCallback(point_cloud): 
    pc = ros_numpy.numpify(point_cloud)
    
    # point_cloud_list_131072 = point_cloud_list_1.reshape([131072, 9])
    # point_cloud_list_1024_128 = point_cloud_list_1.reshape([1024, 128, 9])
    # point_cloud_list_128_1024 = point_cloud_list_1.reshape([128,1024, 9])

    
    marker2= Marker()

    marker2.header.frame_id = "os_sensor"
    marker2.header.stamp = rospy.Time.now()
    
    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker2.type = 2
    marker2.id = 0

    # Set the scale of the marker
    marker2.scale.x = 0.3
    marker2.scale.y = 0.3
    marker2.scale.z = 0.3

    # Set the color
    marker2.color.r = 0.0
    marker2.color.g = 1.0
    marker2.color.b = 0.0
    marker2.color.a = 1.0
    
    # # Set the pose of the marker
    print("rect_listt", rect_list)

    if len(rect_list)>2:
        print(pc.shape)
        print([rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"],])
        print(pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]])
        
        if not pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]][0] == 0 and not pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]][1] == 0:
            marker2.pose.position.x = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]][2]
            # print("marker1.pose.position.x1", marker1.pose.position.x)
        elif not pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]][0] == 0 and not pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]][1]== 0:
            marker2.pose.position.x = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]][2]
            # print("marker1.pose.position.x2", marker1.pose.position.x)
        elif not pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]+1][0] == 0 and not pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]+1][1] == 0:
            marker2.pose.position.x = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]+1][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]+1][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]+1][2]
            # print("marker1.pose.position.x3", marker1.pose.position.x)
        elif not pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]][0] == 0 and not pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]][1] == 0:
            marker2.pose.position.x = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]][2]
            # print("marker1.pose.position.x4", marker1.pose.position.x)
        elif not pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]-1][0] == 0 and not pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]-1][1] == 0:
            marker2.pose.position.x = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]-1][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]-1][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"], rect_list[2]["bd_center_px"]-1][2]
            # print("marker1.pose.position.x5", marker1.pose.position.x)
        elif not pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]+1][0] == 0 and not pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]+1][1] == 0:
            marker2.pose.position.x = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]+1][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]+1][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]+1][2]
            # print("marker1.pose.position.x6", marker1.pose.position.x)
        elif not pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]+1][0] == 0 and not pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]+1][1] == 0:
            marker2.pose.position.x = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]+1][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]+1][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]+1][2]
        elif not pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]-1][0] == 0 and not pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]-1][1] == 0:
            marker2.pose.position.x = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]-1][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]-1][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"]+1, rect_list[2]["bd_center_px"]-1][2]
        elif not pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]-1][0] == 0 and not pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]-1][1] == 0:
            v.pose.position.x = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]-1][0]
            marker2.pose.position.y = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]-1][1]
            marker2.pose.position.z = pc[rect_list[2]["bd_center_py"]-1, rect_list[2]["bd_center_px"]-1][2]
        # else:
        #     marker1.pose.position.x = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+2][0]
        #     marker1.pose.position.y = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+2][1]
        #     marker1.pose.position.z = pc[rect_list[0]["bd_center_y"], rect_list[0]["bd_center_x"]+2][2]
        print("marker1.pose.position", marker2.pose.position)
        # data=np.float_(np.array(marker1.pose.position.x, marker1.pose.position.y))
        # plt.plot(data[0], data[1], linestyle='solid', marker='o', color='red')
        # plt.plot(data[0], data[3], linestyle='solid', marker='o', color='blue')
        # # plt.savefig("")
        # plt.show()

        
    #     # while not rospy.is_shutdown():
        marker_pub.publish(marker2)
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







# topicName_reflec="/ouster/reflec_image"
# topicName_dpt="/ouster/range_image"
# topicName_point="/points"




mf=pub_sub()
# mf.registerCallback(ImageCallback_realsense)
mf.registerCallback(ImageCallback)

rospy.Subscriber("/ouster/points", PointCloud2, PcdCallback)
marker_pub=rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
rospy.spin()
