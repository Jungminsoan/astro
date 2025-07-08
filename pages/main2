import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import asyncio
import platform

# 천구와 지구의 기하학적 설정
R = 1  # 천구의 반지름
tilt = np.radians(23.5)  # 지구 자전축 기울기 (라디안)

# 천구의 구면 좌표 생성
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x = R * np.outer(np.cos(u), np.sin(v))
y = R * np.outer(np.sin(u), np.sin(v))
z = R * np.outer(np.ones(np.size(u)), np.cos(v))

# 별(임의의 점) 추가: 예시로 북극성(Polaris) 근처
stars = np.array([[0, 0, R], [0.1, 0.1, R*0.95]])  # (x, y, z)

# 회전 행렬 (z축 기준, 자전축 기울기 고려)
def rotation_matrix(angle, tilt):
    # 자전축 회전 (z축 -> 기울어진 축)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tilt), -np.sin(tilt)],
                   [0, np.sin(tilt), np.cos(tilt)]])
    # z축 기준 회전
    Rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]])
    return Rx @ Rz @ np.linalg.inv(Rx)

# 초기 설정
def setup():
    global fig, ax
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_zlim(-R*1.2, R*1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Celestial Sphere Rotation (1 Day)')

# 프레임 업데이트
def update_loop(frame):
    ax.clear()
    # 하루 회전: frame은 0~360도
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
    
    # 천구 그리기
    ax.plot_wireframe(x_rot, y_rot, z_rot, color='lightblue', alpha=0.5)
    # 별 그리기
    ax.scatter(stars_rot[:,0], stars_rot[:,1], stars_rot[:,2], color='yellow', s=50)
    # 지구 중심
    ax.scatter([0], [0], [0], color='green', s=100, label='Earth')
    # 자전축 표시
    axis_len = R*1.2
    axis = np.array([[0, 0, -axis_len], [0, 0, axis_len]])
    axis_rot = np.array([rot @ a for a in axis])
    ax.plot(axis_rot[:,0], axis_rot[:,1], axis_rot[:,2], 'r-', label='Rotation Axis')
    
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_zlim(-R*1.2, R*1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f'Celestial Sphere Rotation: {frame:.1f}° (Hour: {frame/15:.1f})')

# 애니메이션 실행
async def main():
    setup()
    ani = FuncAnimation(fig, update_loop, frames=np.linspace(0, 360, 100), interval=50)
    plt.show()
    while True:
        await asyncio.sleep(1.0 / 60)  # 60 FPS

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
