#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


class SubscribePointCloud(object):
    def __init__(self):
        rospy.init_node('subscribe_custom_point_cloud')
        rospy.Subscriber('/custom_point_cloud', PointCloud2, self.callback)
        rospy.spin()

    def callback(self, point_cloud):
        for point in pc2.read_points(point_cloud):
            rospy.logwarn("x, y, z: %.1f, %.1f, %.1f" % (point[0], point[1], point[2]))
            rospy.logwarn("my field 1: %f" % (point[4]))
            rospy.logwarn("my field 2: %f" % (point[5]))


def main():
    try:
        SubscribePointCloud()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

