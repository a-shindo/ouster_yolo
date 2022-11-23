#! /usr/bin/python3
# -*- coding: utf-8 -*-

# launchに組み込むなら1行目にpythonのversionを指定しておくとよい（2で動いちゃう）

import os
import sys
import time
import numpy as np
import cv2
import rospy
import tf
import torch #ない
from pprint import pprint
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped, Pose2D
import message_filters
from cv_bridge import CvBridge

# 個人識別
# from ytlab_whill.ytlab_whill_modules.scripts.reference.individual_recognition_index import personReidentification,Tracker


"""
メモ
・python2/3互換
    ・print -> pprint
・カメラの向きを考慮した座標変換
    ・今の座標がいきあたりばったりすぎるので整理
・座標軸がdepthのpointcloudと一致しない問題
    ・整理して書き直して考える
・rosbagの方も、/camera/aligned_depth_to_color/image_rawを録画してもらえるように設定を変更

ImageCallbackでやること
・深さ・RGB画像・カメラパラメタを取り出す
向きを整える
深さ画像をカラーで表示
深さとして採用した値とマッチするところを深さ画像にも表示
・bounding boxを検出する
・体の座標を出す
・目標値を設定


"""

rotate_mode='right_down' # False,right_down,left_down
dpt_align=True

print(os.getcwd())


class Human_tracking():
    def __init__(self):
        # self.model=torch.hub.load('ultralytics/yolov5','yolov5s')
        self.model = torch.hub.load("/usr/local/lib/python3.8/dist-packages/yolov5", 'custom', path=os.environ['HOME']+'/catkin_ws/src/recog_opencv/config/yolov5/yolov5s.pt',source='local')
        # os.environ['HOME'] ... /home/hayashide_kazuyuki
        # 'path/to/yolov5', 'custom', path='path/to/best.pt', source='local'
        rospy.init_node('human_detection')
        mf=self.pub_sub()
        mf.registerCallback(self.ImageCallback)
        rospy.spin()

    def pub_sub(self):
        sub_list=[]
        topic_prefix="camera3"
        # subscriber
        rgb_sub=message_filters.Subscriber(topic_prefix+"/camera/color/image_raw",Image)
        sub_list.append(rgb_sub)
        if dpt_align:
            dpt_sub=message_filters.Subscriber(topic_prefix+"/camera/aligned_depth_to_color/image_raw",Image)
        else:
            dpt_sub=message_filters.Subscriber(topic_prefix+"/camera/depth/image_rect_raw",Image)
        sub_list.append(dpt_sub)
        info_sub=message_filters.Subscriber(topic_prefix+"/camera/depth/camera_info",CameraInfo)
        sub_list.append(info_sub)
        mf=message_filters.ApproximateTimeSynchronizer(sub_list,100,0.5)

        # publisher
        self.pub_PointStamped=rospy.Publisher("publisher_point",PointStamped,queue_size=1)
        self.pub_Pose2D=rospy.Publisher("publisher_pose",Pose2D,queue_size=1)

        # listener
        self.listener=tf.TransformListener()
        # broadcaster
        return mf
    
    def get_position(self,rgb_array,dpt_array,obj_people,P):
        # alignでない場合のために、縮尺を導出
        y_rgb2dpt=dpt_array.shape[0]/rgb_array.shape[0]
        x_rgb2dpt=dpt_array.shape[1]/rgb_array.shape[1]
        # bounding boxをdpt画像に投射。
        rect_list=[]
        # 識別情報のデータベース
        # identifys = np.zeros((len(obj_people.itertuples()), 255))

        for i,row in enumerate(obj_people.itertuples()):
            xmin_dpt=row.xmin*x_rgb2dpt
            ymin_dpt=row.ymin*y_rgb2dpt
            xmax_dpt=row.xmax*x_rgb2dpt
            ymax_dpt=row.ymax*y_rgb2dpt
            bd_box=np.array(dpt_array[int(ymin_dpt):int(ymax_dpt),int(xmin_dpt):int(xmax_dpt)])
            dpt=np.median(bd_box)
            bd_center_y=int((ymin_dpt+ymax_dpt)/2)
            bd_center_x=int((xmin_dpt+xmax_dpt)/2)
            center_3d=dpt*np.dot(np.linalg.pinv(P),np.array([bd_center_x,bd_center_y,1]).T)
            one_person=[int(xmin_dpt),int(ymin_dpt),int(xmax_dpt),int(ymax_dpt),bd_center_x,bd_center_y,center_3d,dpt]
            rect_list.append(one_person)

            # 個人識別とid付与
            # identifys[i] = personReidentification.infer(bd_box)
        # ids = Tracker.getIds(identifys, rect_list[:,0:4])
        # print(ids)
        

        return rect_list # [xmin,ymin,xmax,ymax,bd_center_x,bd_center_y,center_3d,dpt]
    
    def draw_rect(self,img,rect_list):
        
        for i,person in enumerate(rect_list):
            cv2.rectangle(
                img,
                pt1=(person[0],person[1]),
                pt2=(person[2],person[3]),
                color=(0,0,0)
                )
            cv2.putText(
            img,
            text="P"+str(i)+" : "+str(int(person[-1])),
            org=(person[0],person[1]),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0,0,0),
            thickness=2,
            )
        
        return img
    
    def publish_PointStamped(self,rect_list):
        for person in rect_list:
            point=PointStamped()
            point.header.stamp=rospy.Time.now()
            point.header.frame_id="/camera2_link"
            # point.header.frame_id="/camera_depth_optical_frame" # DepthCloudと比べるとき用
            point.point.x=person[-2][2]/1000
            point.point.y=person[-2][1]/1000
            point.point.z=-person[-2][0]/1000
            """
            カメラから見た移動方向とpointstamped座標の関係
            [x,y,z]=person[-2[0,1,2]のとき
            カメラから見て右方向へ移動 = x増大
            カメラから見て下方向へ移動 = y増大
            カメラから見て奥方向へ移動 = z増大


            """
            self.pub_PointStamped.publish(point)

    def point_translation(self,person):
        return (person[-2][2]/1000, person[-2][1]/1000, -person[-2][0]/1000)

    def publish_Pose2D(self,x,y,theta):
        pose=Pose2D()
        pose.x,pose.y,pose.theta=x,y,theta
        self.pub_Pose2D.publish(pose)
    
    def broadcast_human(self,rect_list):
        for person in rect_list:
            br_human=tf.TransformBroadcaster()
            trans=self.point_translation(person)
            rot=(0,0,0,1)
            try:
                br_human.sendTransform(trans,rot,rospy.Time.now(),"/human","/camera2_link")
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

    def broadcast_goal(self):
        ## vector from base_link to human
        try:
            (trans,rot)=self.listener.lookupTransform("/base_link","/human",rospy.Time(0))

            ## goal pose
            goal_dist=1
            goal_trans=tuple(np.array(trans)-goal_dist)
            goal_rot=rot

            ## publish goal pose
            br=tf.TransformBroadcaster()
            br.sendTransform(goal_trans,goal_rot,rospy.Time.now(),"/goal","/base_link")
            
            # calculate theta
            e = tf.transformations.euler_from_quaternion(goal_rot)
            return goal_trans[0],goal_trans[1],e[2]

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 0,0,0

    def publish_goal(self):
        try:
            (trans,rot)=self.listener.lookupTransform("/map","/human",rospy.Time(0))
            e = tf.transformations.euler_from_quaternion(rot)
            self.publish_Pose2D(trans[0],trans[1],e[2])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        


    def ImageCallback(self,rgb_data,dpt_data,info_data):
        try:
            # unpack arrays
            rgb_array = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(rgb_data.height, rgb_data.width, -1)
            rgb_array=np.nan_to_num(rgb_array)
            rgb_array=cv2.cvtColor(rgb_array,cv2.COLOR_BGR2RGB)
            dpt_array = np.frombuffer(dpt_data.data, dtype=np.uint16).reshape(dpt_data.height, dpt_data.width, -1)
            dpt_array=np.nan_to_num(dpt_array)
            Proj_mtx=np.array(info_data.P).reshape(3,4)

            # rotation
            if rotate_mode=='left_down':
                rotation=cv2.ROTATE_90_COUNTERCLOCKWISE
                rgb_array=cv2.rotate(rgb_array,rotation)
                dpt_array=cv2.rotate(dpt_array,rotation)
            elif rotate_mode=='right_down':
                rotation=cv2.ROTATE_90_CLOCKWISE
                rgb_array=cv2.rotate(rgb_array,rotation)
                dpt_array=cv2.rotate(dpt_array,rotation)
            else:
                pass
            
            # object recognition
            results=self.model(rgb_array)
            objects=results.pandas().xyxy[0]
            obj_people=objects[objects['name']=='person']
            rect_list=self.get_position(rgb_array,dpt_array,obj_people,Proj_mtx)
            
            # publish for visualization
            for person in rect_list:
                # human position
                person_x,person_y,person_z=self.point_translation(person)
                person_theta=0
                self.publish_Pose2D(person_x,person_y,person_theta)

                # goal position
                goal_x,goal_y,goal_theta=self.broadcast_goal()
                self.publish_Pose2D(goal_x,goal_y,goal_theta)

            # calculate goal pose
            self.broadcast_goal()
            self.publish_goal()

            # imshow
            results.render()
            cv2.imshow("detected",results.imgs[0])
            dpt_array_show=(dpt_array-np.min(dpt_array))/(np.max(dpt_array)-np.min(dpt_array))*255
            dpt_array_show=cv2.applyColorMap(np.uint8(dpt_array_show),cv2.COLORMAP_JET)
            dpt_array_show=self.draw_rect(dpt_array_show,rect_list)
            cv2.imshow("depth",dpt_array_show)
            cv2.waitKey(1)


        except Exception:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                pprint(exc_type, fname, exc_tb.tb_lineno)

Human_tracking()


# print("###### DEBUG ROI STARTS ######")
# print("###### DEBUG ROI ENDS ######")