
import numpy as np
import matplotlib.pyplot as plt

ground_truth = np.load("data/ground_truth_positions.npy")
basic = np.load("data/basic_associated_positions.npy")
kalman = np.load("data/kalman_associated_positions.npy")

# Calculate mean 3D position error per frame
basic_errors = np.linalg.norm(
    basic - ground_truth, axis=2
)

kalman_errors = np.linalg.norm(
    kalman - ground_truth, axis=2
)

basic_frame_error = np.mean(basic_errors, axis=1)
kalman_frame_error = np.mean(kalman_errors, axis=1)

frames = np.arange(len(ground_truth))

plt.figure(figsize=(10, 6))

plt.plot(
    frames,
    basic_frame_error,
    marker="o",
    label="Basic Hungarian"
)

plt.plot(
    frames,
    kalman_frame_error,
    marker="s",
    label="Kalman + Hungarian"
)

plt.xlabel("Frame")
plt.ylabel("Mean 3D Position Error")
plt.title("Comparison of Data Association Methods")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/association_method_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Comparison graph generated successfully!")
print("Saved: results/association_method_comparison.png")
