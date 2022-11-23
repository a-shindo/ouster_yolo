# rosbagの画像データを動画 (avi)に変換

import os
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError, core
from sensor_msgs.msg import Image
from glob import glob

mov_save_dir="/mnt/ssd/movie/"

topics = [
        "camera4/camera/color/image_raw/",
        "camera4/camera/aligned_depth_to_color/image_raw",
        # "camera4/camera/depth/image_rect_raw/", 
        ]

topic_name=["color","dpt"]
        
bag_dir = "/mnt/ssd/rosbag"
bags = sorted(glob(bag_dir+"/*"))

bag_basename=[]
for bag in bags:
    bag_basename.append(os.path.splitext(os.path.basename(bag))[0])
    
print(bags)


images_master = [[], [], [], []]

for bag_idx,bag in enumerate(bags):
    for topic_idx, topic in enumerate(topics):
        print(f"now processing: {bag} {topic_name[topic_idx]}")
        for current_topic, current_msg, current_t in rosbag.Bag(bag).read_messages():
            if current_topic == topic:
                try:
                    images_master[topic_idx].append(CvBridge().imgmsg_to_cv2(current_msg, "bgr8"))
                except core.CvBridgeError:
                    images_master[topic_idx].append(CvBridge().imgmsg_to_cv2(current_msg,"8UC1"))

        try:
            print("### len",len(images_master[topic_idx][0]))
        except IndexError:
            print("### no frames found ###")
            continue
        size = (images_master[topic_idx][0].shape[1], images_master[topic_idx][0].shape[0])
        out = cv2.VideoWriter(mov_save_dir+bag_basename[bag_idx]+"_"+topic_name[topic_idx]+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        print(f"now exporting: {mov_save_dir+bag_basename[bag_idx]+'_'+topic_name[topic_idx]+'.avi'}")
        for i in range(len(images_master[topic_idx])):
            out.write(images_master[topic_idx][i])
        out.release()
