
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path="/home/hayashide/kazu_ws/human_recognition/object_detector/csv/stop_go/0902_1_25222.csv"
graph_dir="/home/hayashide/kazu_ws/human_recognition/object_detector/graph/closeup"
csv_data=np.loadtxt(csv_path,delimiter=",")

columns=["now","xmin","ymin","xmax","ymax","bd_center_x","bd_center_y","center_3d_x","center_3d_y","center_3d_z","center_3d_1","confidence","dpt"]
csv_data_pd=pd.DataFrame(csv_data,columns=columns)

ROI_data_time=[]
ROI_data_dist=[]
ROI_start=0
start_append=False
for time_stamp,distance in zip(csv_data_pd["now"],csv_data_pd["dpt"]):
    ROI_data_time.append(time_stamp)
    ROI_data_dist.append(distance)
print(ROI_data_dist)

plt.scatter(ROI_data_time,ROI_data_dist,label=f"distance ({os.path.basename(csv_path)[:-4]})")
plt.title(f"{os.path.basename(csv_path)[:-4]}")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("distance [mm]")

plt.xlim((5,25))
plt.ylim((0,1000))

# plt.show()
plt.savefig(graph_dir+"/"+os.path.basename(csv_path)[:-4]+".jpg",dpi=300)
