# 3D Multi-Object Tracking Using Point Clouds and Kalman Filtering

## Overview

This project is a Python-based prototype for tracking multiple moving
objects in 3D point-cloud data. It uses synthetic point clouds to
demonstrate clustering, centroid extraction, motion prediction, data
association, Kalman filtering, missed-detection handling, track
management, and re-identification.

**Scope:** This is a controlled synthetic experiment. It is not a
validated real-world LiDAR tracking system.

## Objectives

-   Generate synthetic 3D point-cloud sequences for multiple moving
    objects.
-   Detect object clusters using DBSCAN.
-   Estimate object positions from cluster centroids.
-   Predict object motion with a constant-velocity Kalman filter.
-   Associate detections with existing tracks using the Hungarian
    algorithm.
-   Apply distance gating to reject implausible matches.
-   Handle missed detections and maintain track states.
-   Demonstrate dormant-track re-identification.
-   Evaluate position estimates using MAE, RMSE, and mean 3D position
    error.

## Methodology

1.  **Synthetic data generation:** Create noisy 3D point clouds for
    three objects across ten frames.
2.  **Clustering:** Apply DBSCAN to separate point clouds into object
    clusters.
3.  **Centroid extraction:** Calculate one 3D centroid for each detected
    object.
4.  **Motion prediction:** Use a constant-velocity Kalman filter with
    state \[ \[x, y, z, v_x, v_y, v_z\]\^T. \]
5.  **Data association:** Use predicted track positions, Euclidean
    distances, and the Hungarian assignment algorithm.
6.  **Distance gating:** Reject candidate associations beyond a
    configured distance threshold.
7.  **Track management:** Mark tracks as active or dormant, and remove
    tracks after the configured missed-frame limit.
8.  **Re-identification:** Compare new detections with archived dormant
    tracks and restore an earlier identity when the distance criterion
    is satisfied.
9.  **Evaluation:** Compare estimated positions with the known synthetic
    ground truth.

## Experimental Setup

-   Number of objects: 3
-   Number of frames: 10
-   Points per object per frame: 100
-   Points per frame: 300
-   Point-cloud noise: Gaussian, standard deviation 0.4
-   Clustering: DBSCAN (`eps=1.0`, `min_samples=5`)
-   Tracking state: 3D position and 3D velocity
-   Association: Hungarian algorithm
-   Example distance-gating threshold: 2.0
-   Re-identification threshold: 2.5
-   Long-disappearance test: Object 1 is hidden for frames 3--7 and
    returns in frame 8

The thresholds are experimental settings for this synthetic scene and
should not be interpreted as universally optimal values.

## Selected Results

### Association comparison

  Method                                  MAE       RMSE   Mean 3D position error
  -------------------------------- ---------- ---------- ------------------------
  Basic Hungarian association        0.031838   0.038478                 0.061342
  Kalman + Hungarian association     0.029093   0.036183                 0.056996

In this experiment, Kalman-based association produced lower position
errors than basic Hungarian association. Both methods preserved
identities in this relatively simple, well-separated synthetic scene.

### Missed-detection experiment

  Experiment                                         MAE       RMSE
  ------------------------------------------- ---------- ----------
  Standard Kalman tracking                      0.029093   0.036183
  Tracking with simulated missed detections     0.028657   0.036809

The missed-detection experiment removed selected measurements. The small
change in metrics is specific to this short synthetic sequence; a lower
MAE in one run does not mean missed detections improve tracking.

### Re-identification experiment

  Metric                       Result
  ------------------------ ----------
  MAE                        0.051987
  RMSE                       0.094510
  Mean 3D position error     0.105735

The prototype successfully restored the original identity of the
returning object after its five-frame disappearance. Only one controlled
re-identification event was tested, so this result is illustrative
rather than a general performance guarantee.

During the dormant period, position error increased because the filter
predicted without receiving measurements. The returning detection
corrected the estimate when the track was re-identified.

## Main Technologies

-   Python
-   NumPy
-   Matplotlib
-   SciPy
-   scikit-learn

## Installation

Clone the repository and move into the project directory:

``` bash
git clone https://github.com/Majid123-web/3d-multi-object-tracking.git
cd 3d-multi-object-tracking
```

Create and activate a virtual environment (Windows PowerShell):

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Running the Project

Run scripts from the project root directory. For example:

``` bash
python src/kalman_tracking.py
python src/kalman_data_association.py
python src/kalman_missed_detection_tracking.py
python src/track_reidentification.py
python src/evaluate_reidentification_accuracy.py
```

Some scripts depend on data files generated by earlier scripts. Follow
the generation and processing order used by the project; the commands
above are examples, not a guaranteed standalone sequence.

## Repository Structure

The following is a representative structure. The exact files may vary as
the project develops.

``` text
3d-multi-object-tracking/
├── data/        # Generated point clouds, centroids, and tracking arrays
├── src/         # Data generation, clustering, tracking, and evaluation scripts
├── results/     # Plots, figures, and result summaries
├── models/      # Model-related files, if used
├── README.md
└── requirements.txt
```

## Limitations

-   Experiments use synthetic data rather than real sensor or LiDAR
    recordings.
-   The scene contains only three relatively well-separated objects.
-   The sequence is short (ten frames).
-   Distance and re-identification thresholds are configured for this
    experiment.
-   Re-identification is demonstrated with one controlled disappearance
    case.
-   Performance on dense scenes, heavy occlusion, complex motion, and
    real-world sensor noise has not been established.
-   Standard multi-object tracking metrics such as MOTA, IDF1, and HOTA
    have not been implemented.

## Future Work

-   Test on real-world 3D LiDAR or point-cloud datasets.
-   Evaluate more objects, longer sequences, and overlapping
    trajectories.
-   Improve motion and noise models.
-   Add track-level metrics such as identity switches, MOTA, IDF1, and
    HOTA.
-   Study robustness under occlusion, clutter, and varying point
    density.

## Author

**Majid Khan**

GitHub: https://github.com/Majid123-web
