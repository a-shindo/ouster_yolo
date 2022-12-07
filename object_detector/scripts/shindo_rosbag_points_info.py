"""
https://gist.github.com/yuma-m/fbc5d8ef8882b1eeb264

"""

import rospy
from sensor_msgs.msg import PointCloud2  # Topicの型に合わせて引用
#import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
#import ros_numpy
#from cv_bridge import CvBridge

def callback(point_cloud):  # 呼び出し関数
    #rospy.loginfo(pixelValue)  # pythonのprint
    #rospy.loginfo(str(pixelValue))
    # point_cloud_list = pc2.read_points_list(point_cloud)
    # point_cloud_list_1 = np.array(point_cloud_list)
    #print(point_cloud_list_1.shape)
    # point_cloud_list_128_1024 = point_cloud_list_1.reshape([128, 1024, 9])
    
    #print("len(point_cloud_list)", len(point_cloud_list))
    # print("point_cloud_list_128_1024", point_cloud_list_128_1024)
    for point in pc2.read_points_list(point_cloud):
       rospy.loginfo(point)
       print(point)
    #print(point_cloud.height)
    #print(point_cloud.width)
    
    
    
def listener():
    rospy.init_node('listener')  # ('Node名')
    rospy.Subscriber("/ouster/points", PointCloud2, callback)  # ("Topic名", 型, 関数)
    rospy.spin()

if __name__ == '__main__':
    listener()



