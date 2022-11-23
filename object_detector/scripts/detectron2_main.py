import os
from glob import glob
import subprocess as sp

from detectron2_core import *
"""
model type
OD: object detection
IS: instance segmentation
LVIS: LVinstance segmentation
PS: panoptic segmentation
KP: keypoint detection
"""
detector=Detector(model_type="KP")

# results=detector.onImage(imagePath="/home/hayashide/catkin_ws/src/object_detector/images/sources/00_no_lost.jpeg")
# print(list(detector.onImage(imagePath="/home/hayashide/catkin_ws/src/object_detector/images/sources/00_no_lost.jpeg")))#[0].numpy())
# print(results)

videos=sorted(glob("/home/hayashide/catkin_ws/src/object_detector/scripts/temp/sources/*"))


for videoPath in videos:
    video_basename=os.path.basename(videoPath)
    if video_basename[-4:]==".mp4":
        detector.onVideo(videoPath=videoPath,savePath=f"/home/hayashide/catkin_ws/src/object_detector/scripts/temp/results/{video_basename}",csvPath=f'/home/hayashide/catkin_ws/src/object_detector/csv/{video_basename[:-4]}.csv')

######### 途中までやったなら、そこをスキップするのをお忘れなく！！！！！