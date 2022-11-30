#!/usr/bin/env python
# coding: UTF-8
"""
import numpy as np
import rosbag
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def camera_callback(msg):
    try:
        bridge = CvBridge()
        cv_array = bridge.imgmsg_to_cv2(msg)
        rospy.loginfo(cv_array)
 
    except Exception, err:
        rospy.logerr(err)

if __name__ == '__main__':
    rospy.init_node('special_node', log_level=rospy.DEBUG)
    rospy.Subscriber('ouster/range_image', Image, camera_callback)
"""
import rospy
from sensor_msgs.msg import Image  # Topicの型に合わせて引用

def callback(data):  # 呼び出し関数
    rospy.loginfo(data.data)  # pythonのprint
    
def listener():
    rospy.init_node('listener')  # ('Node名')
    rospy.Subscriber("/ouster/range_image", Image, callback)  # ("Topic名", 型, 関数)
    rospy.spin()

if __name__ == '__main__':
    listener()

