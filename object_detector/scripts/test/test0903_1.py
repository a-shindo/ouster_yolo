import os
from glob import glob

bag_dir="/mnt/ssd/rosbag"
bags=glob(bag_dir+"/*")
bag_basename=[]
for bag in bags:
    bag_basename.append(os.path.splitext(os.path.basename(bag))[0])

["xmin","ymin","xmax","ymax","bd_center_x","bd_center_y","center_3d","confidence","dpt"]    
print(bag_basename)
