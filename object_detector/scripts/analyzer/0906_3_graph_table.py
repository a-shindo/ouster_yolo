import numpy as np
import cv2
from glob import glob

graph_master_dir="/home/hayashide_kazuyuki/ytlab_ros_ws/ytlab_yolov5/object_detector/graph/all"
table_graph_dir="/home/hayashide_kazuyuki/ytlab_ros_ws/ytlab_yolov5/object_detector/graph/table"

dir_paths=sorted(glob(graph_master_dir+"/*"))
dir_paths=dir_paths[:5]

column_num=len(dir_paths)
row_num=len(sorted(glob(dir_paths[0]+"/*")))
print(row_num,column_num)

(graph_row_size,graph_column_size,color_num)=cv2.imread(sorted(glob(dir_paths[0]+"/*"))[0]).shape
print(graph_row_size,graph_column_size,color_num)

canvas=np.zeros((graph_row_size*row_num,graph_column_size*column_num,color_num))

for column,dir_path in enumerate(dir_paths):
    graph_paths=sorted(glob(dir_path+"/*"))
    for row,graph_path in enumerate(graph_paths):
        graph=cv2.imread(graph_path)
        canvas[row*graph_row_size:(row+1)*graph_row_size,column*graph_column_size:(column+1)*graph_column_size,:]=graph
canvas=cv2.resize(canvas,(int(canvas.shape[1]/10),int(canvas.shape[0]/10)))
cv2.imwrite(table_graph_dir+"/table_all.jpg",canvas)
