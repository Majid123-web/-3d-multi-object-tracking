
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load saved tracking history
history = np.load("data/track_identity_history.npy")

# Define status colors
colors = [
    "lightgray",  # 0: Not created
    "green",      # 1: Active
    "orange",     # 2: Dormant
    "blue"        # 3: Re-identified
]

cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

fig, ax = plt.subplots(figsize=(12, 5))

image = ax.imshow(
    history.T,
    cmap=cmap,
    norm=norm,
    aspect="auto"
)

# Axis labels
ax.set_xlabel("Frame Number", fontsize=12)
ax.set_ylabel("Track ID", fontsize=12)
ax.set_title("Multi-Object Track Identity History", fontsize=14)

ax.set_xticks(range(history.shape[0]))
ax.set_yticks(range(history.shape[1]))
ax.set_yticklabels(
    [f"Track {i+1}" for i in range(history.shape[1])]
)

# Add status numbers inside cells
for track in range(history.shape[1]):
    for frame in range(history.shape[0]):

        ax.text(
            frame,
            track,
            str(history[frame, track]),
            ha="center",
            va="center",
            color="black",
            fontsize=10
        )

# Create legend
from matplotlib.patches import Patch

legend_items = [
    Patch(facecolor="lightgray", label="Not Created"),
    Patch(facecolor="green", label="Active"),
    Patch(facecolor="orange", label="Dormant"),
    Patch(facecolor="blue", label="Re-identified")
]

ax.legend(
    handles=legend_items,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4
)

plt.tight_layout()

plt.savefig(
    "results/track_identity_history.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Identity history visualization saved!")
print("File: results/track_identity_history.png")
