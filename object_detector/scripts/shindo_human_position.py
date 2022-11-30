import csv
import numpy as np


csv_path = f"/home/ytpc2017d/catkin_ws/src/object_detector/csv/_2022-10-25-17-35-13.csv"

rows = []
with open(csv_path) as f:   
    reader = csv.reader(f)
    rows = [row for row in reader]

data = np.float_(np.array(rows).T)
# リストのデータを、横軸用のデータ、縦軸用のデータに直すために、np.array(rows)でnumpy型に直した後に、「.T」で転置
# このままだとデータが文字列型になっているので「np.float_( )」で浮動小数点に直す

#バウンディングボックスの中心点の決定
box_center_x1= (data[1]+data[3])/2
box_center_x= np.floor(box_center_x1) #小数点以下切り捨て
box_center_y1= (data[2]+data[4])/2
box_center_y= np.floor(box_center_y1)
print(box_center_x, box_center_y)

center_3d=
