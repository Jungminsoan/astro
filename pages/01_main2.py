import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 천구 설정
R = 1  # 천구 반지름
tilt = np.radians(23.5)  # 지구 자전축 기울기

# 천구 좌표 생성
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))

# 별 좌표 (예: 북극성 근처)
stars = np.array([[0, 0, R], [0.1, 0.1, R * 0.95]])

# 회전 행렬 (z축 기준, 자전축 기울기 고려)
def rotation_matrix(angle, tilt):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]])
    return Rx @ Rz @ np.linalg.inv(Rx)

# 플롯 설정
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def init():
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_zlim(-R*1.2, R*1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Celestial Sphere Rotation')
    return ax,

def update(frame):
    ax.clear()
    angle = np.radians(frame)
    rot = rotation_matrix(angle, tilt)
    
    # 천구 회전
    x_rot = np.zeros_like(x)
    y_rot = np.zeros_like(y)
    z_rot = np.zeros_like(z)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vec = np.array([x[i,j], y[i,j], z[i,j]])
            vec_rot = rot @ vec
            x_rot[i,j], y_rot[i,j], z_rot[i,j] = vec_rot
    
    # 별 회전
    stars_rot = np.array([rot @ star for star in stars])
    
    # 플롯
    ax.plot_wireframe(x_rot, y_rot, z_rot, color='lightblue', alpha=0.5)
    ax.scatter(stars_rot[:,0], stars_rot[:,1], stars_rot[:,2], color='yellow', s=50)
    ax.scatter([0], [0], [0], color='green', s=100, label='Earth')
    axis_len = R * 1.2
    axis = np.array([[0, 0, -axis_len], [0, 0, axis_len]])
    axis_rot = np.array([rot @ a for a in axis])
    ax.plot(axis_rot[:,0], axis_rot[:,1], axis_rot[:,2], 'r-', label='Rotation Axis')
    
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_zlim(-R*1.2, R*1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Celestial Sphere: Hour {frame/15:.1f} ({frame:.1f}°)')
    ax.legend()
    return ax,

# 애니메이션 생성
ani = FuncAnimation(fig, update, init_func=init, frames=np.linspace(0, 360, 100), interval=50, blit=False)

# 깃허브용: 애니메이션 저장 (로컬 테스트 시 주석 해제)
# ani.save('celestial_sphere.mp4', writer='ffmpeg', fps=20)

# 화면 표시
plt.show()
