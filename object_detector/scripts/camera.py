#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
realsenseを通常のRGBカメラとして使用するためのコード。
スマホ等だと画角・解像度が本番環境と異なるため、実験関係はすべてrealsenseで完結させたいという狙い。

用意したいパラメータ
- カメラの名前
- 向きの回転
- タイマー
"""
import os
import time
import cv2
import rospy
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from cv_bridge import CvBridge

camera_frame="camera3"
que=False
s_time=1e100


def subs():
    sub_list=[]
    # subscriber
    rgb_sub=message_filters.Subscriber(camera_frame+"/camera/color/image_raw",Image)
    sub_list.append(rgb_sub)

    dpt_sub=message_filters.Subscriber(camera_frame+"/camera/aligned_depth_to_color/image_raw",Image)

    sub_list.append(dpt_sub)

    info_sub=message_filters.Subscriber(camera_frame+"/camera/depth/camera_info",CameraInfo)
    sub_list.append(info_sub)

    mf=message_filters.ApproximateTimeSynchronizer(sub_list,100,0.5)

    return mf

def ImageCallback(rgb_data,dpt_data,info_data):
    print(time.time())
    try:
        bridge = CvBridge()
        cv_array = bridge.imgmsg_to_cv2(rgb_data,"bgr8")
        cv2.imshow("rgb",cv_array)
        cv2.waitKey(1)

    except Exception as err:
        rospy.logerr(err)

    global que
    global s_time
    key=cv2.waitKey(1) & 0xFF
    if key ==ord("c"):
        print("take picture")
        s_time=time.time()
        # cv2.imwrite("rgb.jpg",cv_array)
        que=True
    if key == ord("q"):
        cv2.destroyAllWindows()
    if que and time.time()-s_time>=timer:
        cv2.imwrite("~/catkin_ws/src/object_detector/images/rgb.jpg",cv_array)
        cv2.destroyAllWindows()

    

os.system("source ~/.bashrc")
os.system("bash ~/devel/setup.bash")
timer=float(input("Input wait time in sec. :"))
print("Press 'c' to take a picture. Press 'q' to quit.")
rospy.init_node('rs_ctrl')
mf=subs()
mf.registerCallback(ImageCallback)
rospy.spin()
