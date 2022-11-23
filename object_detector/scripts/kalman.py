import pandas as pd
import numpy as np

dt=1/30
A=np.array([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0],
    [0,0,0,1]
])
P=np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
])
W=np.array([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
])
V=np.array([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
])

for i in range(1,3):
    print(f"###### SEQUENCE {i} #####")
    P=np.dot(np.dot(A,P),np.linalg.pinv(A))-np.dot(np.dot(np.dot(np.dot(A,P),np.linalg.pinv(P+V)),P),A)+W
    H=np.dot(np.dot(A,P),np.linalg.pinv(P+V))
    print("P",P)
    print("H",H)

csv_path="/home/hayashide/kazu_ws/human_recognition/object_detector/csv/1004_detectron2.csv"
csv_data=np.loadtxt(csv_path,delimiter=",")
csv_data_pd=pd.DataFrame(csv_data).to_numpy()
print(csv_data_pd)

data_old=None
x=np.array([
        [data[1]],
        [data[2]],
        [data_old[1]*(1+dt)],
        [data_old[1]*(1+dt)],

])
for i,data in enumerate(csv_data_pd):
    if np.isnan(data[2]):
        print(f"###### SEQUENCE {i+2} #####")
        x=np.array([
            [data[1]],
            [data[2]],
            [data_old[1]*(1+dt)],
            [data_old[1]*(1+dt)],
        ])
    data_old=data
    pass