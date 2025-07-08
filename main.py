# main.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Streamlit 페이지 설정
st.set_page_config(page_title="천구 기하학 시뮬레이션", layout="wide")
st.title("천구 기하학적 시뮬레이션")

# 천구 데이터 생성
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# 천구 좌표를 직교 좌표로 변환
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Matplotlib 3D 플롯 생성
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.6)
ax.set_xlabel("X 축")
ax.set_ylabel("Y 축")
ax.set_zlabel("Z 축")
ax.set_title("3D 천구 시뮬레이션")

# Streamlit에 플롯 표시
st.pyplot(fig)

# 추가 설명
st.write("이 그래프는 천구를 3D로 시각화한 것입니다. X, Y, Z 축으로 표현된 구형 좌표계입니다.")
