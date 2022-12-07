"""
https://gist.github.com/yuma-m/fbc5d8ef8882b1eeb264

"""

import rospy
from sensor_msgs.msg import PointCloud2  # Topicの型に合わせて引用
#import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import re
#import ros_numpy
#from cv_bridge import CvBridge

def callback(point_cloud):  # 呼び出し関数
    # rospy.loginfo(pixelValue)  # pythonのprint
    # rospy.loginfo(str(pixelValue))
    point_cloud_list = pc2.read_points_list(point_cloud)
    # print("point_cloud_list", point_cloud_list)
    # print("len(point_cloud_list)", len(point_cloud_list))

    point_cloud_list_1 = np.array(point_cloud_list)
    # print(point_cloud_list_1.shape)
    point_cloud_list_131072 = point_cloud_list_1.reshape([131072, 9])
    point_cloud_list_1024_128 = point_cloud_list_1.reshape([1024, 128, 9])
    point_cloud_list_128_1024 = point_cloud_list_1.reshape([128,1024, 9])
    # point_cloud_list_9 = point_cloud_list_131072[131071, :]
    # point_cloud_list_ = point_cloud_list_128_1024[:, :]
    point_cloud_list_1 = point_cloud_list_131072[129, :]
    point_cloud_list_2_1 = point_cloud_list_1024_128[1, 0,:]
    # point_cloud_list_2_2 = point_cloud_list_1024_128[0, 129,:]
    point_cloud_list_3_1 = point_cloud_list_128_1024[1, 0, :]
    point_cloud_list_3_2 = point_cloud_list_128_1024[0, 129, :]
    point_cloud_list_0_0 = point_cloud_list_128_1024[0, 0, :]
    point_cloud_list_0_1023 = point_cloud_list_128_1024[0, 1023, :]
    point_cloud_list_127_0 = point_cloud_list_128_1024[127, 0, :]
    point_cloud_list_127_1023 = point_cloud_list_128_1024[127, 1023, :]

    # print("point_cloud_list_128_1024", point_cloud_list_128_1024)
    print("\npoint_cloud_list_1", point_cloud_list_1)
    # print("point_cloud_list_2_1", point_cloud_list_2_1)
    # print("point_cloud_list_2_2", point_cloud_list_2_2)
    # print("point_cloud_list_3_1", point_cloud_list_3_1)
    print("point_cloud_list_3_2", point_cloud_list_3_2)
    print("point_cloud_list_0_0", point_cloud_list_0_0)
    print("point_cloud_list_0_1023", point_cloud_list_0_1023)
    print("point_cloud_list_127_0", point_cloud_list_127_0)
    print("point_cloud_list_127_1023", point_cloud_list_127_1023)    
    
    
    
def listener():
    rospy.init_node('listener')  # ('Node名')
    rospy.Subscriber("/ouster/points", PointCloud2, callback)  # ("Topic名", 型, 関数)
    rospy.spin()

if __name__ == '__main__':
    listener()



