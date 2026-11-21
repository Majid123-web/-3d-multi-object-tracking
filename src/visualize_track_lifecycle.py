
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Track status:
# 0 = Not created
# 1 = Active (measurement received)
# 2 = Missed detection (prediction only)
# 3 = Deleted

frames = np.arange(10)

track_data = np.array([
    [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],  # Track 1
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Track 2
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Track 3
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]   # Track 4
])

colors = [
    "lightgray",
    "green",
    "orange",
    "red"
]

cmap = ListedColormap(colors)

plt.figure(figsize=(12, 5))

plt.imshow(
    track_data,
    cmap=cmap,
    aspect="auto",
    interpolation="nearest",
    vmin=0,
    vmax=3
)

plt.xticks(frames)
plt.yticks(
    range(4),
    ["Track 1", "Track 2", "Track 3", "Track 4"]
)

plt.xlabel("Frame Number")
plt.ylabel("Track ID")

plt.title("Multi-Object Tracking: Track Lifecycle")

legend_items = [
    Patch(color="lightgray", label="Not Created"),
    Patch(color="green", label="Active"),
    Patch(color="orange", label="Missed Detection"),
    Patch(color="red", label="Deleted")
]

plt.legend(
    handles=legend_items,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4
)

plt.grid(
    axis="x",
    color="black",
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    "results/track_lifecycle.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Track lifecycle visualization completed!")
print("Saved: results/track_lifecycle.png")
