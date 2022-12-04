import rospy
from sensor_msgs.msg import Image  # Topicの型に合わせて引用
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import ros_numpy

def callback(data):  # 呼び出し関数
    rospy.loginfo(data.data, dtype=np.float32)  # pythonのprint
    # rospy.loginfo(np.array(pixelValue))
    #img = ("/ouster/range_image")
    #pixelValue = img[10, 20]
    
def listener():
    rospy.init_node('listener')  # ('Node名')
    rospy.Subscriber("/ouster/range_image", Image, callback)  # ("Topic名", 型, 関数)
    rospy.spin()

if __name__ == '__main__':
    listener()



