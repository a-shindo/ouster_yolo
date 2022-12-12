import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = f"/home/ytpc2017d/catkin_ws/src/object_detector/csv2/xy.csv"

# for data in csv_path:
#     plt.scatter(data[0], data[1])
#     plt.ylabel('expenses')
#     plt.xlabel('smoke')
#     plt.show()

input_csv = pd.read_csv('/home/ytpc2017d/catkin_ws/src/object_detector/csv2/xy.csv')
first_column_data = input_csv[input_csv.keys()[0]]
second_column_data = input_csv[input_csv.keys()[1]]

plt.xlabel(input_csv.keys()[0])
plt.ylabel(input_csv.keys()[1])
# plt.scatter(first_column_data, second_column_data, linestyle='solid', marker='o')
plt.plot(first_column_data, second_column_data, linestyle='solid', marker='o')
plt.ylabel('y')
plt.xlabel('x')
plt.show()
plt.savefig("/home/ytpc2017d/object_detector/graph/x_y.png")
