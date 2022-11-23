import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

dt=1/30
A=np.array([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0],
    [0,0,0,1]
],dtype=np.float64)
P=np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
],dtype=np.float64)
W=np.random.randn(4,4)
V=np.random.randn(4,4)

for i in range(1,3):
    print(f"###### SEQUENCE {i-2} #####")
    W=np.random.randn(4,4)
    V=np.random.randn(4,4)
    P=np.dot(np.dot(A,P),np.linalg.pinv(A))-np.dot(np.dot(np.dot(np.dot(A,P),np.linalg.pinv(P+V)),P),A.T)+W
    H=np.dot(np.dot(A,P),np.linalg.pinv(P+V))
    print("P",P)
    print("H",H)

# ここまででP0,H0が求まった。

csv_path="/home/hayashide/kazu_ws/human_recognition/object_detector/csv/1004_detectron2.csv"
csv_path=np.loadtxt(csv_path,delimiter=",")
csv_data=pd.DataFrame(csv_path).to_numpy()
print(csv_data)

new_csv_path="/home/hayashide/kazu_ws/human_recognition/object_detector/csv/1004_detectron2_kalman.csv"

# comp_flg=False
# new_csv_data=[]
# for i,data in enumerate(csv_data):
#     print(f"###### SEQUENCE {i} #####")
#     if i==0:
#         data_old=np.zeros_like(csv_data[0])
#         Xhat=np.zeros((4,1))
#     else:
#         data_old=csv_data[i-1]

#     try:
#         data_next=csv_data[i+1]
#     except IndexError:
#         break
#     P=np.dot(np.dot(A,P),np.linalg.pinv(A))-np.dot(np.dot(np.dot(np.dot(A,P),np.linalg.pinv(P+V)),P),A.T)+W
#     H=np.dot(np.dot(A,P),np.linalg.pinv(P+V))
#     X=np.array([data[1],data[2],(data[1]-data_old[1])/dt,(data[2]-data_old[2])/dt,]).reshape(-1,1)
#     if np.isnan(X[0]) or np.isnan(X[1]) or np.isnan(X[2]) or np.isnan(X[3]):
#         print("alternative")
#         X=np.array([new_csv_data[-1][1],new_csv_data[-1][2],(new_csv_data[-1][1]-new_csv_data[-2][1])/dt,(new_csv_data[-1][2]-new_csv_data[-2][2])/dt])
#     else:
#         X=np.array([data[1],data[2],(data[1]-data_old[1])/dt,(data[2]-data_old[2])/dt,]).reshape(-1,1)

#     if np.isnan(X[0]) or np.isnan(X[1]) or np.isnan(X[2]) or np.isnan(X[3]):
#         print("!!!!!!!!!!!!!!!!!!!!!")
        
    
#     Xhat=np.dot(A,Xhat)+np.dot(H,X-Xhat)

#     new_data=copy.deepcopy(data)
#     np.put(new_data,[1,2],[Xhat[0],Xhat[1]])

#     new_csv_data.append(new_data)
#     # print(Xhat)
#     np.savetxt(new_csv_path,new_csv_data,delimiter=",")


kalman_data=np.loadtxt(new_csv_path,delimiter=",")
kalman_data=pd.DataFrame(kalman_data).to_numpy()

plt.plot(csv_data[:,1],csv_data[:,2],label="raw")
plt.plot(kalman_data[:,1],kalman_data[:,2],label="kalman")
plt.xlim((0,2000))
plt.ylim((0,1000))
plt.legend()
plt.show()







"""
    if comp_flg:
        comp_flg=False
        pass
    else:
        X=np.array([
                data[1],
                data[2],
                (data[1]-data_old[1])/dt,
                (data[2]-data_old[2])/dt,
        ],dtype=np.float64).T
    
    if i==0:
        Xhat=X
    
    if np.isnan(data_next[1]) or np.isnan(data[1]) or np.isnan(data_old[1]):
        print(f"######SEQUENCE {i} X[{i}] -> X[{i+1}] #####")
        P=np.dot(np.dot(A,P),np.linalg.pinv(A))-np.dot(np.dot(np.dot(np.dot(A,P),np.linalg.pinv(P+V)),P),A.T)+W
        H=np.dot(np.dot(A,P),np.linalg.pinv(P+V))
        print("P",P)
        print("H",H)
        print("X",X)
        Xhat=np.dot(A,Xhat)+np.dot(H,X-Xhat)
        print("Xhat",Xhat)
        X=Xhat
        comp_flg=True


        data_old=data
    history.append([X[0],X[1]])
    pass
history=np.array(history)
print(np.array(history)[:,0])
plt.plot(history[:,0],history[:,1],label="kalman")
plt.plot(csv_data[:,1],csv_data[:,2],label="raw")
plt.legend()
plt.show()"""