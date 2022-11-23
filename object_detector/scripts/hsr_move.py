#!/usr/bin/python
# -*- coding: utf-8 -*-

from turtle import pu
import rospy
from hsrb_interface import Robot
import controller_manager_msgs.srv
import trajectory_msgs.msg
import numpy as np

rospy.init_node("hsr_move")
language = rospy.get_param("speak_language", "Japanese")
robot = Robot()
base = robot.try_get('omni_base')
tts = robot.try_get('default_tts')
whole_body = robot.try_get('whole_body')
# gripper=robot.try_get('gripper')

# go straight
def pub_prepare():
    pub = rospy.Publisher(
        '/hsrb/omni_base_controller/command',
        trajectory_msgs.msg.JointTrajectory, queue_size=10)

    # wait to establish connection between the controller
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)

    # make sure the controller is running
    rospy.wait_for_service('/hsrb/controller_manager/list_controllers')
    list_controllers = rospy.ServiceProxy(
        '/hsrb/controller_manager/list_controllers',
        controller_manager_msgs.srv.ListControllers)
    running = False
    while running is False:
        rospy.sleep(0.1)
        for c in list_controllers().controller:
            if c.name == 'omni_base_controller' and c.state == 'running':
                running = True
    return pub

# fill ROS message
def run(positions=[0,0,0],duration=15):
    pub=pub_prepare()

    traj = trajectory_msgs.msg.JointTrajectory()
    traj.joint_names = ["odom_x", "odom_y", "odom_t"]
    p = trajectory_msgs.msg.JointTrajectoryPoint()
    p.positions = positions # front(abs), left(abs), counter-clock(rel)
    p.velocities = [0, 0, 0]
    p.time_from_start = rospy.Duration(duration)
    traj.points = [p]

    # publish ROS message
    pub.publish(traj)

run([0,0,0],duration=5)
rospy.sleep(5)
duration=30
run([3,0,0],duration=duration)
rospy.sleep(duration)
run([0,0,0],duration=duration)
rospy.sleep(duration)
