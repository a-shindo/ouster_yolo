"""
https://answers.ros.org/question/373802/minimal-working-example-for-rviz-marker-publishing/

"""
import rospy
from visualization_msgs.msg import Marker
import rospy
from sensor_msgs.msg import PointCloud2  
import numpy as np
import sensor_msgs.point_cloud2 as pc2


class PcdListener:
    def __init__(self) -> None:
        rospy.init_node('listener')  # ('Node名')
        rospy.Subscriber("/ouster/points", PointCloud2, self.callback)  # ("Topic名", 型, 関数)
        self.marker_pub=rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        print("init node")
        rospy.spin()
    def callback(self, point_cloud):  
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

        # marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 2)
        
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
        # marker2.color.r = 1.0
        # marker2.color.g = 0.0
        # marker2.color.b = 0.0
        # marker2.color.a = 1.0

        # Set the pose of the marker
        # marker1.pose.position.x = point_cloud_list_1024_128[400, 1, 0]
        # marker1.pose.position.y = point_cloud_list_1024_128[400, 1, 1]
        # marker1.pose.position.z = point_cloud_list_1024_128[400, 1, 2]
        marker1.pose.position.x = point_cloud_list_128_1024[100, 600, 0]
        marker1.pose.position.y = point_cloud_list_128_1024[100, 600, 1]
        marker1.pose.position.z = point_cloud_list_128_1024[100, 600, 2]
        print("point_cloud_list_128_1024[2, 1, :]", point_cloud_list_128_1024[0, 1, :])
        print("point_cloud_list_128_1024[2, 1, 0]", point_cloud_list_128_1024[2, 1, 0])
        print("point_cloud_list_128_1024[2, 1, 1]", point_cloud_list_128_1024[2, 1, 1])
        print("point_cloud_list_128_1024[2, 1, 2]", point_cloud_list_128_1024[2, 1, 2])
        print("marker.pose.position.x", marker1.pose.position)
        # marker.pose.orientation.x = 0.0
        # marker.pose.orientation.y = 0.0
        # marker.pose.orientation.z = 0.0
        # marker.pose.orientation.w = 1.0

        # while not rospy.is_shutdown():
        self.marker_pub.publish(marker1)
        # rospy.rostime.wallsleep(1.0)    




if __name__ == '__main__':
    listener=PcdListener()



