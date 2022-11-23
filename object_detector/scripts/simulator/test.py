import numpy as np
import matplotlib.pyplot as plt

def U_wall(subject_position,center_position,):
    # 上に凸なポテンシャル場（→斥力）の計算式
    param_x=1.0
    param_y=1.0
    potential=param_x*(subject_position[0]-center_position[0])**2+param_y*(subject_position[1]-center_position[1])**2
    return potential

def W_human(subject_position,center_position,):
    # 下に凸なポテンシャル場（→引力）の計算式
    param_x=1.0
    param_y=1.0
    potential=-param_x*(subject_position[0]-center_position[0])**2+-param_y*(subject_position[1]-center_position[1])**2
    return potential

def potential_for_heatmap(X,Y):
    wall1_y=0.0
    wall2_y=3.0

    pot_wall1=U_wall((X,Y),(X,np.full_like(X,float(wall1_y))))
    pot_wall2=U_wall((X,Y),(X,np.full_like(X,float(wall2_y))))
    pot_human=W_human((X,Y),(10,2.5))
    return pot_wall1+pot_wall2+pot_human
    pass




x=np.linspace(0,20,200)
y=np.linspace(-1,4,50)
X, Y = np.meshgrid(x, y)
print(X)
Z=potential_for_heatmap(X,Y)

fig, ax = plt.subplots()
contf = ax.contourf(X, Y, Z, 10, cmap='PuOr')
ax.set_aspect('equal','box')
plt.colorbar(contf)
plt.show()