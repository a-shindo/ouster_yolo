import rospy
from sensor_msgs.msg import Image  # Topicの型に合わせて引用
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
from cv_bridge import CvBridge

def callback(pixelValue):  # 呼び出し関数
    #rospy.loginfo(pixelValue)  # pythonのprint
    #rospy.loginfo(str(pixelValue))
    img = CvBridge().imgmsg_to_cv2(pixelValue)
    pixelValue = img[10, 20]
    rospy.loginfo(img)
    #img = ("/ouster/range_image")
    
    
def listener():
    rospy.init_node('listener')  # ('Node名')
    rospy.Subscriber("/ouster/range_image", Image, callback)  # ("Topic名", 型, 関数)
    rospy.spin()

if __name__ == '__main__':
    listener()



