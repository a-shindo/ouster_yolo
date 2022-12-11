import csv
import numpy as np
import matplotlib.pyplot as plt

csv_path = f"/home/ytpc2017d/catkin_ws/src/object_detector/csv2/xy.csv"

for data in csv_path:
    plt.scatter(data[0], data[1])
    plt.ylabel('expenses')
    plt.xlabel('smoke')
    plt.show()
