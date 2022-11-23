import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

csv_path="/home/hayashide_kazuyuki/ytlab_ros_ws/ytlab_yolov5/object_detector/csv/0904_point.csv"
graph_dir="/home/hayashide_kazuyuki/ytlab_ros_ws/ytlab_yolov5/object_detector/graph"

csv_data=np.loadtxt(csv_path,delimiter=",")
columns=["x","y","z"]
csv_data_pd=pd.DataFrame(csv_data,columns=columns)

plt.scatter(np.arange(0,len(csv_data_pd["x"])),csv_data_pd["x"],s=3,label="x")
plt.scatter(np.arange(0,len(csv_data_pd["y"])),csv_data_pd["y"],s=3,label="y")
plt.scatter(np.arange(0,len(csv_data_pd["z"])),csv_data_pd["z"],s=3,label="z")
plt.legend()

plt.savefig(graph_dir+"/0905_point.jpg")
