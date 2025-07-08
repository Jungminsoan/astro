import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# 오늘 날짜 (2025-07-08) 기준 항성시 계산
def get_initial_angle(date):
    # 기준: 2000년 1월 1일 정오 UT의 항성시 0도
    # 2025년 7월 8일까지의 경과 일수
    epoch = datetime(2000, 1, 1, 12, 0, 0)
    current = datetime(2025, 7, 8, 12, 0, 0)  # 오늘 12:00 KST
    days_since_epoch = (current - epoch).total_seconds() / (24 * 3600)
    # 항성일(23h 56m 4s)로 회전 각도 계산
    sidereal_day = 23.9344696 / 24  # 항성일/태양일 비율
    rotation_angle = (days_since_epoch * 360 * sidereal_day) % 360
    return np.radians(rotation_angle)

# 천구 설정
R = 1  # 천구 반지름
tilt = np.radians(23.5)  # 지구 자전축 기울기

# 천구 좌표 생성
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.swapaxes(0, 1))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))

# 별 좌표 (북극성 등)
stars = np.array([[0, 0, R], [0.1, 0.1, R * 0.95]])

# 회전 행렬
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

# 초기 각도 (2025-07-08 기준)
initial_angle = get_initial_angle(datetime.now())
angle = initial_angle

# 천구 회전
rot = rotation_matrix(angle, tilt)
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
ax.scatter(stars_rot[:,0], stars_rot[:,1], stars_rot[:,2], color='yellow', s=50, label='Stars')
ax.scatter([0], [0], [0], color='green', s=100, label='Earth')
axis_len = R * 1.2
axis = np.array([[0, 0, -axis_len], [0, 0, axis_len]])
axis_rot = np.array([rot @ a for a in axis])
ax.plot(axis_rot[:,0], axis_rot[:,1], axis_rot[:,2], 'r-', label='Rotation Axis')

# 축 설정
ax.set_xlim(-R*1.2, R*1.2)
ax.set_ylim(-R*1.2, R*1.2)
ax.set_zlim(-R*1.2, R*1.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Celestial Sphere on 2025-07-08 (Angle: {(angle * 180 / np.pi):.1f}°)')
ax.legend()

# 정적 플롯 표시
plt.show()

# 애니메이션 (필요 시 주석 해제)
"""
from matplotlib.animation import FuncAnimation

def update(frame):
    ax.clear()
    angle = initial_angle + np.radians(frame)
    rot = rotation_matrix(angle, tilt)
    
    x_rot = np.zeros_like(x)
    y_rot = np.zeros_like(y)
    z_rot = np.zeros_like(z)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vec = np.array([x[i,j], y[i,j], z[i,j]])
            vec_rot = rot @ vec
            x_rot[i,j], y_rot[i,j], z_rot[i,j] = vec_rot
    
    stars_rot = np.array([rot @ star for star in stars])
    
    ax.plot_wireframe(x_rot, y_rot, z_rot, color='lightblue', alpha=0.5)
    ax.scatter(stars_rot[:,0], stars_rot[:,1], stars_rot[:,2], color='yellow', s=50, label='Stars')
    ax.scatter([0], [0], [0], color='green', s=100, label='Earth')
    axis_rot = np.array([rot @ a for a in axis])
    ax.plot(axis_rot[:,0], axis_rot[:,1], axis_rot[:,2], 'r-', label='Rotation Axis')
    
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_zlim(-R*1.2, R*1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Celestial Sphere on 2025-07-08: Hour {frame/15:.1f} ({frame:.1f}°)')
    ax.legend()
    return ax,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 360, 100), interval=50, blit=False)
plt.show()
# ani.save('celestial_sphere_20250708.mp4', writer='ffmpeg', fps=20)  # MP4 저장 (FFmpeg 필요)
"""
