#! /usr/bin/python3
# -*- coding: utf-8 -*-

import rospy

def main():
    rospy.init_node('mesureing_computation_time', anonymous=True)

    t_start = rospy.get_time()

    ## Start
    for i in range(10000):
        print("i = ", i)
        print("time[s] = ", rospy.get_time() - t_start)
    ## End

    print("Computation is done")
    print("Computation time[s] = ", rospy.get_time() - t_start)

if __name__ == '__main__':
    main()