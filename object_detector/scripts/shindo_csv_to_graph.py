"""
https://python-academia.com/matplotlib-csv/

時間の整形
https://qiita.com/srs/items/4d19a749891728c8520a
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = f"/home/ytpc2017d/catkin_ws/src/object_detector/csv/_2022-10-25-17-35-13.csv"

rows = []
with open(csv_path) as f:   
    reader = csv.reader(f)
    rows = [row for row in reader]

#header = rows.pop(0)

data = np.float_(np.array(rows).T)

# リストのデータを、横軸用のデータ、縦軸用のデータに直すために、np.array(rows)でnumpy型に直した後に、「.T」で転置
# このままだとデータが文字列型になっているので「np.float_( )」で浮動小数点に直す

#plt.xlabel(header[0])
#plt.ylabel(header[2])

plt.plot(data[0], data[1], linestyle='solid', marker='o', color='red')
plt.plot(data[0], data[3], linestyle='solid', marker='o', color='blue')
plt.xlim(0, max(data[0]))
plt.ylim(0, 1024)
# plt.savefig("_2022-10-25-17-35-13_x.png")
plt.show()
