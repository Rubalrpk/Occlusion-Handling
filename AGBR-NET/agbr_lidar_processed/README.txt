AGBR-Net LiDAR dataset processing output

Source: five ROS 2 SQLite bag recordings supplied by the user.
The /scan topic is sensor_msgs/msg/LaserScan in frame 'laser_frame'.

Files per scene:
  *_raw_laserscan.csv  - all valid LiDAR rays with scan index, timestamps, angle, range, intensity.
  *_median_scan.csv     - stationary-scene median range summarized into 1-degree angular bins.
  *_lidar_map.png       - 2D Cartesian map from the median scan.

dataset_summary.csv gives recording-level statistics.

Important alignment note:
The RGB camera was reported to be at the same position/height and pointing in the same forward direction as the LiDAR. For pixel-level RGB/LiDAR projection, camera horizontal field-of-view or intrinsic calibration is still required.
