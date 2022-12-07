"""
https://answers.ros.org/question/373802/minimal-working-example-for-rviz-marker-publishing/

"""
import rospy
from visualization_msgs.msg import Marker
import rospy
from sensor_msgs.msg import PointCloud2  
import numpy as np
import sensor_msgs.point_cloud2 as pc2


def pub_sub():
    global rgb_sub,dpt_sub,info_sub
    # subscriber
    sub_list=[]
    rgb_sub=message_filters.Subscriber(topicName_rgb,Image)
    sub_list.append(rgb_sub)
    dpt_sub=message_filters.Subscriber(topicName_dpt,Image)
    sub_list.append(dpt_sub)
    info_sub=message_filters.Subscriber(topicName_camInfo,CameraInfo)
    sub_list.append(info_sub)
    mf=message_filters.ApproximateTimeSynchronizer(sub_list,10,0.5)

def callback(point_cloud):  
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

    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)

    marker = Marker()

    marker.header.frame_id = "os_sensor"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = 2
    marker.id = 0

    # Set the scale of the marker
    marker.scale.x = 0.3
    marker.scale.y = 0.3
    marker.scale.z = 0.3

    # Set the color
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Set the pose of the marker
    marker.pose.position.x = point_cloud_list_128_1024[127, 1023, 0]
    marker.pose.position.y = point_cloud_list_128_1024[127, 1023, 1]
    marker.pose.position.z = point_cloud_list_128_1024[127, 1023, 2]
    # marker.pose.orientation.x = 0.0
    # marker.pose.orientation.y = 0.0
    # marker.pose.orientation.z = 0.0
    # marker.pose.orientation.w = 1.0

    # while not rospy.is_shutdown():
    # marker_pub.publish(marker)
    # rospy.rostime.wallsleep(1.0)    



def listener():
    rospy.init_node('listener')  # ('Node名')
    rospy.Subscriber("/ouster/points", PointCloud2, callback)  # ("Topic名", 型, 関数)
    rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
    rospy.spin()


