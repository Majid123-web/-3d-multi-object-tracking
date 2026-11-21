
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Step 1: Load the saved dataset
all_frames = np.load("data/pointcloud_frames.npy")

# Step 2: Select Frame 0
pointcloud = all_frames[0]

print("Frame 0 loaded successfully!")
print("Number of points:", pointcloud.shape[0])

# Step 3: Apply DBSCAN clustering
clustering = DBSCAN(
    eps=1.0,
    min_samples=5
)

labels = clustering.fit_predict(pointcloud)

# Step 4: Count detected clusters
unique_labels = np.unique(labels)

num_clusters = len(unique_labels[unique_labels != -1])
num_noise = np.sum(labels == -1)

print("Number of detected clusters:", num_clusters)
print("Number of noise points:", num_noise)
# Step 5: Calculate the centroid of each cluster
print("\nObject Centers (Centroids):")

for label in unique_labels:

    if label == -1:
        continue

    cluster_points = pointcloud[labels == label]

    centroid = np.mean(cluster_points, axis=0)

    print(f"Object {label + 1}:")
    print("X:", centroid[0])
    print("Y:", centroid[1])
    print("Z:", centroid[2])

# Step 5: Visualize the clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

for label in unique_labels:

    cluster_points = pointcloud[labels == label]

    if label == -1:
        color = "black"
        name = "Noise"
    else:
        color = plt.cm.tab10(label % 10)
        name = f"Cluster {label + 1}"

    ax.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        cluster_points[:, 2],
        color=color,
        label=name,
        s=12
    )

ax.set_title("DBSCAN Clustering of 3D Point Cloud")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()
