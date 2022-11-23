from turtle import color
import numpy as np

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob



csv_dir="/home/hayashide_kazuyuki/ytlab_ros_ws/ytlab_yolov5/object_detector/csv"
ROI_distances=[1000,2000,3000,6000,9000,12000,15000,18000,21000,24000,27000,30000]
for ROI_distance in ROI_distances:
    graph_dir=f"/home/hayashide_kazuyuki/ytlab_ros_ws/ytlab_yolov5/object_detector/graph/all/{str(ROI_distance).zfill(5)}mm"
    try:
        os.mkdir(graph_dir)
    except FileExistsError:
        pass

    csv_paths=sorted(glob(csv_dir+"/*"))

    # ROI_distance=3000 # mm
    ROI_time=80/60*48
    distance_tolerance=500 # mm
    minimum_points=20

    for csv_path in csv_paths:
        print(csv_path)
        try:
            csv_data=np.loadtxt(csv_path,delimiter=",")
        except IsADirectoryError:
            continue
        columns=["now","xmin","ymin","xmax","ymax","bd_center_x","bd_center_y","center_3d_x","center_3d_y","center_3d_z","center_3d_1","confidence","dpt"]
        csv_data_pd=pd.DataFrame(csv_data,columns=columns)

        ROI_data_time=[]
        ROI_data_dist=[]
        ROI_start=0
        start_append=False
        for time_stamp,distance in zip(csv_data_pd["now"],csv_data_pd["dpt"]):
            if distance > ROI_distance-distance_tolerance and distance < ROI_distance+distance_tolerance and time_stamp-ROI_start<ROI_time:
                ROI_data_time.append(time_stamp)
                ROI_data_dist.append(distance)
                start_append=True
            elif not(start_append) or len(ROI_data_time)<minimum_points:
                ROI_start=time_stamp
            else:
                break
        
        plt.scatter(ROI_data_time,ROI_data_dist,label=f"distance ({os.path.basename(csv_path)[:-4]})")
        plt.title(f"{os.path.basename(csv_path)[:-4]}")
        plt.plot(ROI_data_time,np.full_like(ROI_data_time,ROI_distance),label="true distance",color="red")
        plt.legend()
        plt.xlabel("time [s]")
        plt.ylabel("distance [mm]")
        # plt.show()
        plt.savefig(graph_dir+"/"+os.path.basename(csv_path)[:-4]+".jpg",dpi=300)
        plt.cla()

    for csv_path in csv_paths:
        print(csv_path)
        try:
            csv_data=np.loadtxt(csv_path,delimiter=",")
        except IsADirectoryError:
            continue
        columns=["now","xmin","ymin","xmax","ymax","bd_center_x","bd_center_y","center_3d_x","center_3d_y","center_3d_z","center_3d_1","confidence","dpt"]
        csv_data_pd=pd.DataFrame(csv_data,columns=columns)

        ROI_data_time=[]
        ROI_data_dist=[]
        ROI_start=0
        start_append=False
        for time_stamp,distance in zip(csv_data_pd["now"],csv_data_pd["dpt"]):
            if distance > ROI_distance-distance_tolerance and distance < ROI_distance+distance_tolerance and time_stamp-ROI_start<ROI_time:
                ROI_data_time.append(time_stamp)
                ROI_data_dist.append(distance)
                start_append=True
            elif not(start_append):
                ROI_start=time_stamp
            else:
                break
        
        plt.scatter(ROI_data_time,ROI_data_dist,label=f"distance ({os.path.basename(csv_path)[7:-4]})")
        plt.title("compare")
    plt.plot(ROI_data_time,np.full_like(ROI_data_dist,ROI_distance),label="true distance",color="red")
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("distance [mm]")
    plt.savefig(graph_dir+"/"+"all"+".jpg",dpi=300)
    plt.cla()
        # plt.show()
