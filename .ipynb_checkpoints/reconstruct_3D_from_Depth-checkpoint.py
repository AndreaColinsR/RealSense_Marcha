import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp

data = pd.read_csv("./Profundidad/test_3D.csv")
POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)

n_frames = len(data)

# ---------------------------------------
# Compute fixed axis limits
# ---------------------------------------
coords = []

for frame in range(150,250):
    for i in range(33):

        X = data.loc[frame, f"X_{i}"]
        Y = data.loc[frame, f"Y_{i}"]
        Z = data.loc[frame, f"Z_{i}"]

        if np.isnan(X) or np.isnan(Y) or np.isnan(Z):
            continue

        Y = -Y
        coords.append([X, Z, Y])

coords = np.array(coords)

mins = coords.min(axis=0)
maxs = coords.max(axis=0)

center = (mins + maxs) / 2
center[0]=0.6
center[1]=0.5
center[2]=0.2
radius = np.max(maxs - mins) * 0.3
print(radius)

# ---------------------------------------
# Video writer
# ---------------------------------------

width = 800
height = 800
fps = 30

video = cv2.VideoWriter(
    "pose_animation.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

# ---------------------------------------
# Generate each frame
# ---------------------------------------
fig = plt.figure(figsize=(8, 8), dpi=100)
ax = fig.add_subplot(111, projection="3d")

for frame in range(n_frames):

    

    points = {}

    for i in range(33):

        X = data.loc[frame, f"X_{i}"]
        Y = data.loc[frame, f"Y_{i}"]
        Z = data.loc[frame, f"Z_{i}"]

        if np.isnan(X) or np.isnan(Y) or np.isnan(Z):
            continue

        Y = -Y
        points[i] = (X, Z, Y)

    # Draw joints
    for i, (x, z, y) in points.items():
        ax.scatter(x, z, y, color="green", s=30)
        #ax.text(x, z, y, str(i), fontsize=8)

    # Draw bones
    for start, end in POSE_CONNECTIONS:
        if start not in points or end not in points:
            continue

        x1, z1, y1 = points[start]
        x2, z2, y2 = points[end]

        ax.plot(
            [x1, x2],
            [z1, z2],
            [y1, y2],
            color="blue",
            linewidth=2,
        )

    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

    ax.set_xlabel("X")
    ax.set_ylabel("Depth")
    ax.set_zlabel("Height")

    ax.view_init(elev=20, azim=-40)

    # Convert matplotlib figure to OpenCV image
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    #img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    #video.write(img)
    plt.pause(0.05)
    ax.cla()

    print(f"Frame {frame+1}/{n_frames}")

video.release()

print("Video saved!")