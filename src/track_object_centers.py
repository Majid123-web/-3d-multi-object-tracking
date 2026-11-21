
import numpy as np
from sklearn.cluster import DBSCAN

# Step 1: Load all frames
all_frames = np.load("data/pointcloud_frames.npy")

num_frames = all_frames.shape[0]

# Store object centers for every frame
all_centroids = []

# Step 2: Process each frame
for frame_index in range(num_frames):

    pointcloud = all_frames[frame_index]

    # Apply DBSCAN
    clustering = DBSCAN(
        eps=1.0,
        min_samples=5
    )

    labels = clustering.fit_predict(pointcloud)

    unique_labels = np.unique(labels)

    frame_centroids = []

    # Step 3: Calculate each cluster's centroid
    for label in unique_labels:

        if label == -1:
            continue

        cluster_points = pointcloud[labels == label]

        centroid = np.mean(cluster_points, axis=0)

        frame_centroids.append(centroid)

    all_centroids.append(frame_centroids)

    print(
        f"Frame {frame_index}: "
        f"{len(frame_centroids)} objects detected"
    )

print("\nCentroid calculation completed!")

# Step 4: Display the object centers
for frame_index, centroids in enumerate(all_centroids):

    print(f"\nFrame {frame_index}:")

    for object_index, center in enumerate(centroids):

        print(
            f"Object {object_index + 1}: "
            f"X={center[0]:.3f}, "
            f"Y={center[1]:.3f}, "
            f"Z={center[2]:.3f}"
        )

# Step 5: Save centroid data
np.save(
    "data/all_centroids.npy",
    np.array(all_centroids)
)

print("\nCentroids saved to data/all_centroids.npy")
