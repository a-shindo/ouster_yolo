import os
import subprocess as sp
from glob import glob

bags=sorted(glob("/media/hayashide/KIOXIA/hayashide/rosbag/*"))[1:]
avi_path="/home/hayashide/catkin_ws/src/object_detector/scripts/temp/sources/"

for bag in bags:
    bag_basename=os.path.basename(bag)
    cmd=f"roslaunch object_detector conv_bag_to_avi.launch bag_path:={bag} save_path:={avi_path}{bag_basename[:-4]}.avi"
    # runcmd=sp.call(cmd.split())
    # print(runcmd)
    os.system(cmd)