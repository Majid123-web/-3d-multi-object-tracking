
import csv
import os

results = [
    ["Experiment", "Metric", "Result"],
    
    ["Basic Kalman Tracking", "MAE", 0.029093],
    ["Basic Kalman Tracking", "RMSE", 0.036183],
    
    ["Basic Hungarian Association", "MAE", 0.031838],
    ["Basic Hungarian Association", "RMSE", 0.038478],
    
    ["Kalman + Hungarian Association", "MAE", 0.029093],
    ["Kalman + Hungarian Association", "RMSE", 0.036183],
    
    ["Missed Detection Tracking", "MAE", 0.028657],
    ["Missed Detection Tracking", "RMSE", 0.036809],
    
    ["Re-identification", "MAE", 0.051987],
    ["Re-identification", "RMSE", 0.094510],
    ["Re-identification", "Mean 3D Error", 0.105735],
    
    ["Active Tracking", "Mean 3D Error", 0.058722],
    ["Dormant Tracking", "Mean 3D Error", 0.346927],
    ["Re-identified Tracking", "Mean 3D Error", 0.028071],
]

os.makedirs("results", exist_ok=True)

output_file = "results/final_results_summary.csv"

with open(output_file, "w", newline="") as file:

    writer = csv.writer(file)
    writer.writerows(results)

print("Final results summary created!")
print("Saved to:", output_file)

print("\nTotal experiment metrics:", len(results) - 1)
