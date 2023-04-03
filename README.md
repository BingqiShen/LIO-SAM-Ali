# LIO-SAM-Ali

## Dependency

This is the original ROS1 implementation of LIO-SAM. For a ROS2 implementation see branch `ros2`.

- [ROS](http://wiki.ros.org/ROS/Installation) (tested with Kinetic and Melodic. Refer to [#206](https://github.com/TixiaoShan/LIO-SAM/issues/206) for Noetic)
  ```
  sudo apt-get install -y ros-kinetic-navigation
  sudo apt-get install -y ros-kinetic-robot-localization
  sudo apt-get install -y ros-kinetic-robot-state-publisher
  ```
- [gtsam](https://gtsam.org/get_started/) (Georgia Tech Smoothing and Mapping library)
  ```
  sudo add-apt-repository ppa:borglab/gtsam-release-4.0
  sudo apt install libgtsam-dev libgtsam-unstable-dev
  ```

## Install

Use the following commands to download and compile the package.

```
cd ~/catkin_ws/src
git clone https://github.com/TixiaoShan/LIO-SAM.git
cd ..
catkin_make
```

## Using Docker
Build image (based on ROS1 Kinetic):

```bash
docker build -t liosam-kinetic-xenial .
```

Once you have the image, start a container as follows:

```bash
docker run --init -it -d \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  liosam-kinetic-xenial \
  bash
```

## Run the package

1. Run the launch file:
```
roslaunch lio_sam run.launch
```

It's worth mentioning that the topics and frame ids of rtk and imu have been revised by tools/repub.cpp

2. Play existing bag files:
```
rosbag play your-bag.bag -r 3 --pause --clock
```

3. Align GPS and odom coordinates
```
rosrun lio_sam lio_sam_gps_odom_align
```

Here we select 50 samples of rtk and odom output to achieve the alignment via Eigen::Umeyama in tools/gps_odom_align.cpp.

The number of samples can be revised in line 131, and we recommend to start this node when the odom output is stable.

## Service
  - /lio_sam/save_map
    - save map as a PCD file.
      ``` bash
        rosservice call [service] [resolution] [destination]
      ```
      - Example:
      ``` bash
        $ rosservice call /lio_sam/save_map 0.2 "/Downloads/LOAM/"
      ```

## Issues

  - **Zigzag or jerking behavior**: if your lidar and IMU data formats are consistent with the requirement of LIO-SAM, this problem is likely caused by un-synced timestamp of lidar and IMU data.

  - **Jumpping up and down**: if you start testing your bag file and the base_link starts to jump up and down immediately, it is likely your IMU extrinsics are wrong. For example, the gravity acceleration has negative value.

  - **mapOptimization crash**: it is usually caused by GTSAM. Please install the GTSAM specified in the README.md. More similar issues can be found [here](https://github.com/TixiaoShan/LIO-SAM/issues).

  - **gps odometry unavailable**: it is generally caused due to unavailable transform between message frame_ids and robot frame_id (for example: transform should be available from "imu_frame_id" and "gps_frame_id" to "base_link" frame. Please read the Robot Localization documentation found [here](http://docs.ros.org/en/melodic/api/robot_localization/html/preparing_sensor_data.html).

## Paper

Thank you for citing [LIO-SAM (IROS-2020)](./config/doc/paper.pdf) if you use any of this code.
```
@inproceedings{liosam2020shan,
  title={LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping},
  author={Shan, Tixiao and Englot, Brendan and Meyers, Drew and Wang, Wei and Ratti, Carlo and Rus Daniela},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5135-5142},
  year={2020},
  organization={IEEE}
}
```

Part of the code is adapted from [LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM).
```
@inproceedings{legoloam2018shan,
  title={LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain},
  author={Shan, Tixiao and Englot, Brendan},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4758-4765},
  year={2018},
  organization={IEEE}
}
```