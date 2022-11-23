import os
import shutil
import time
import datetime
from glob import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FDM():
    """
    環境に関する定義を入れる。
    """
    def __init__(self):
        self.wall1_y=0.0
        self.wall2_y=3.0

        # results/imagesを初期化
        self.temp_images_dir="/home/hayashide/kazu_ws/human_recognition/object_detector/scripts/simulator/results/images"
        shutil.rmtree(self.temp_images_dir)
        os.mkdir(self.temp_images_dir)
        self.frame_idx=0

        self.save_path=f"/home/hayashide/kazu_ws/human_recognition/object_detector/scripts/simulator/results/movies/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"

        pass

    def U(self,subject_position,center_position,):
        # 上に凸なポテンシャル場（→斥力）の計算式
        param_x=0.1
        param_y=0.1
        potential=-param_x*(subject_position[0]-center_position[0])**2-param_y*(subject_position[1]-center_position[1])**2
        return potential

    def W(self,subject_position,center_position,):
        # 下に凸なポテンシャル場（→引力）の計算式
        param_x=0.05
        param_y=0.05
        potential=param_x*(subject_position[0]-center_position[0])**2+param_y*(subject_position[1]-center_position[1])**2
        return potential




class Human(FDM):
    """
    人間位置・行動に関する変数を管理
    """
    def __init__(self):
        super().__init__()
        human_initial_position=np.array([10.0,2.5,np.pi]).reshape(-1,1)
        self.human_position=human_initial_position
    pass

class Robot(FDM):
    """
    ロボット・ターゲット地点の位置・行動に関する変数を管理
    """
    def __init__(self):
        super().__init__()
        self.target_distance=3.0

        robot_initial_position=np.array([0.5,0.5,0]).reshape(-1,1)
        self.robot_position=robot_initial_position
        self.target_position=self.get_target_position(self.robot_position)

    def get_target_position(self,robot_position):
        x_T=robot_position[0]+self.target_distance*np.cos(robot_position[2])
        y_T=robot_position[1]+self.target_distance*np.sin(robot_position[2])

        return np.array([float(x_T),float(y_T),np.nan]).reshape(-1,1)
    pass

class Goal(FDM):
    def __init__(self):
        super().__init__()

        self.goal_position=np.array([20,0.5,np.nan]).reshape(-1,1)

class Visualization(Human,Robot,Goal):
    """
    根っこのクラスで定義されたことを図示
    """
    def __init__(self):
        Human.__init__(self)
        Robot.__init__(self)
        Goal.__init__(self)
        pass
    def plot_current_situation(self):
        self.fig = plt.figure()
        self.ax = plt.axes()

        xlim=(0,20)
        ylim=(-1,4)

        # potential drawing
        x=np.linspace(0,20,200)
        y=np.linspace(-1,4,50)
        X, Y = np.meshgrid(x, y)
        contf = self.ax.contourf(X, Y, self.potential_for_heatmap(X,Y), 10, cmap='PuOr')
        self.ax.set_aspect('equal','box')
        plt.colorbar(contf)

        # character drawing
        self.plot_character(self.robot_position,"blue")
        self.plot_character(self.human_position,"red")
        self.plot_character(self.target_position,"green")
        self.plot_character(self.goal_position,"purple")

        self.plot_wall()

        # force drawing
        
        plt.axis('scaled')
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        plt.savefig(self.temp_images_dir+f"/{str(self.frame_idx).zfill(4)}.jpg")
        self.frame_idx+=1
        del self.fig
        del self.ax
        pass

    def plot_character(self,position,color):
        if np.isnan(position[2]):
            plt.plot([position[0]-0.25,position[0]+0.25],[position[1]-0.25,position[1]+0.25],c=color)
            plt.plot([position[0]-0.25,position[0]+0.25],[position[1]+0.25,position[1]-0.25],c=color)
            pass
        else: # 有向の場合
            r=0.5
            c = patches.Circle(xy=(position[0], position[1]), radius=r, fc='w', ec=color) # fill color, edge color
            self.ax.add_patch(c)
            plt.plot([position[0],position[0]+r*np.cos(position[2])],[position[1],position[1]+r*np.sin(position[2])])
            # self.ax.annotate('', 
            #                     xy=(position[0]+np.cos(position[2]),position[1]+np.sin(position[2])), 
            #                     xytext=(position[0],position[1]),
            #         arrowprops=dict(shrink=0, width=1, headwidth=8, 
            #                         headlength=10, connectionstyle='arc3',
            #                         facecolor=color, edgecolor=color)
            #    )
    
    def plot_wall(self):

        plt.plot([0,20],[self.wall1_y,self.wall1_y],c="black",lw=0.3)
        plt.plot([0,20],[self.wall2_y,self.wall2_y],c="black",lw=0.3)

    def potential_for_heatmap(self,X,Y):
        pot_wall1=self.U((X,Y),(X,np.full_like(X,float(self.wall1_y))))
        pot_wall2=self.U((X,Y),(X,np.full_like(X,float(self.wall2_y))))
        pot_human=self.W((X,Y),(self.human_position))
        pot_goal=self.W((X,Y),(self.goal_position))
        return pot_wall1+pot_wall2+pot_human+pot_goal

    def create_movies(self):
        image_paths=sorted(glob(self.temp_images_dir+"/*"))
        print(image_paths)
        fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
        print(cv2.imread(image_paths[0]).shape[:2])
        video=cv2.VideoWriter(self.save_path,fourcc, 5.0,(cv2.imread(image_paths[0]).shape[1],cv2.imread(image_paths[0]).shape[0]))
        for image_path in image_paths:
            image=cv2.imread(image_path)
            video.write(image)
            
        video.release()
        

class Controller(Visualization):
    """
    根っこのクラスで定義されたことを組み合わせて、シミュレーションを実行
    ここで新しい環境情報を定義してはいけない（図示に反映できないから）
    """
    def __init__(self):
        super().__init__()
        pass


    def renew_position(self):
        pass

    def test_walk(self):
        self.human_position+=np.array([-0.1,0,0]).reshape(-1,1)
        self.target_position=self.get_target_position(self.robot_position)
        pass
    pass
    



cont=Controller()

for i in range (10):
    cont.test_walk()
    cont.plot_current_situation()
cont.create_movies()