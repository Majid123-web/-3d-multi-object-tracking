
import numpy as np

# Load track identity history
history = np.load("data/track_identity_history.npy")

status_names = {
    0: "Not Created",
    1: "Active",
    2: "Dormant",
    3: "Re-identified"
}

print("TRACK IDENTITY HISTORY EVALUATION")
print("=" * 45)

for track_id in range(history.shape[1]):

    track_status = history[:, track_id]

    print(f"\nTrack {track_id + 1}")

    for status, name in status_names.items():

        count = np.sum(track_status == status)

        print(f"{name}: {count} frames")

    active_frames = np.sum(track_status == 1)
    dormant_frames = np.sum(track_status == 2)
    reidentified_frames = np.sum(track_status == 3)

    print("Total recorded frames:", len(track_status))

print("\n" + "=" * 45)
print("Evaluation completed!")
