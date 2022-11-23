import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

csv_dir="/home/hayashide_kazuyuki/ytlab_ros_ws/ytlab_yolov5/object_detector/csv"
graph_dir="/home/hayashide_kazuyuki/ytlab_ros_ws/ytlab_yolov5/object_detector/graph/whole_process"

csv_paths=sorted(glob(csv_dir+"/*"))
print(csv_paths)

for csv_path in csv_paths:
    try:
        csv_data=np.loadtxt(csv_path,delimiter=",")
    except IsADirectoryError:
        continue
    columns=["now","xmin","ymin","xmax","ymax","bd_center_x","bd_center_y","center_3d_x","center_3d_y","center_3d_z","center_3d_1","confidence","dpt"]
    csv_data_pd=pd.DataFrame(csv_data,columns=columns)
    print(csv_data_pd['dpt'])

    fig, ax1 = plt.subplots()

    ax1.scatter(csv_data_pd['now'],csv_data_pd['dpt']/1000,s=3,label="dpt")

    ax2 = ax1.twinx()
    ax2.scatter(csv_data_pd['now'],csv_data_pd['confidence'],c="red",s=3,label="confidence")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("depth [m]")
    ax1.set_title(f"depth & confidence in exp. {os.path.splitext(os.path.basename(csv_path))[0]}")
    ax2.set_ylabel("confidence")

    graph_path=graph_dir+"/"+os.path.splitext(os.path.basename(csv_path))[0]+".jpg"
    fig.savefig(graph_path,dpi=300)

