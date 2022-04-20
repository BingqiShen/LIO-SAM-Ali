/*
 * Copyright 2021 <copyright holder> <email>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef PREHANDLEMSGS_H
#define PREHANDLEMSGS_H
#include "image_projection.h"

#include <string>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include "imu_tracker.h"
#include "pose_extrapolator/imu_based_pose_extrapolator.h"
#include <geometry_msgs/PoseStamped.h>
class PreHandleMsgs : public ParamServer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PreHandleMsgs(const ros::NodeHandle& n);
  ~PreHandleMsgs();

private:
  void lidarCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg);
  void lidarsCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar1_msg,
                      const sensor_msgs::PointCloud2::ConstPtr& lidar2_msg);
  void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg);
  void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg);
  void preOdomHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg);
  void handlePose(const geometry_msgs::PoseStampedConstPtr& msg);
  bool deskewInfo();
  void cachePointCloud(sensor_msgs::PointCloud2& laserCloudMsg,
                       pcl::PointCloud<PointXYZIRT>::Ptr lidar_cloud,
                       double& end_time, const double& det_time);
  void imuDeskewInfo();
  void odomDeskewInfo();
  void findRotation(double pointTime, float* rotXCur, float* rotYCur,
                    float* rotZCur);
  void findPosition(double relTime, float* posXCur, float* posYCur,
                    float* posZCur);
  void predictPointCloudPoses(const std::string& frame_id,
                              const std::set<float>& points_times);
  void resetParameters();
  std::mutex imu_lock_;
  std::mutex odoLock;

  ros::NodeHandle nh_;
  ros::Subscriber subLaserCloud;
  ros::Publisher pubLaserCloud;

  ros::Publisher pubExtractedCloud;
  ros::Publisher pubLaserCloudInfo;
  ros::Subscriber pose_sub_;
  ros::Subscriber sub_odom_;
  ros::Subscriber sub_imu_;
  ros::Subscriber sub_pre_odom_;

  ros::Subscriber subLaserCloudInfo;

  ros::Publisher pubCornerPoints;
  ros::Publisher pubSurfacePoints;
  tf2_ros::Buffer tfBuffer_;
  tf2_ros::TransformListener tfListener_;

  message_filters::Subscriber<sensor_msgs::PointCloud2> lidar1_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> lidar2_sub_;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> lidarsSyncPolicy;
  message_filters::Synchronizer<lidarsSyncPolicy>* lidarsSync_;

  bool use_lidars_;

  std::unique_ptr<ImageProjection> image_projection_;
  unsigned int size_of_lidar_sensors_;

  std::shared_ptr<ImuTracker> imu_tracker_;
  bool initialised_;
  std::deque<sensor_msgs::Imu> imuQueue;
  std::deque<nav_msgs::Odometry> odomQueue;
  std::map<std::string, std::deque<sensor_msgs::PointCloud2>> cloud_queue_;

  pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
  pcl::PointCloud<HesaiPointXYZIRT>::Ptr tmpHesaiCloudIn;
  std::shared_ptr<lio_sam::cloud_info> cloud_info_;
  std::set<float> points_times_;
  std::unordered_map<double, Eigen::Affine3f> points_poses_;
  std::map<std::string, Eigen::Affine3f> lidars2base_;
  double imuTime[2000];
  double imuRotX[2000];
  double imuRotY[2000];
  double imuRotZ[2000];

  bool odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  int imuPointerCur;
  int deskewFlag;
  double cur_time_;
  double cur_start_time_;

  //   std::unique_ptr<PoseExtrapolatorInterface> pose_extrapolator_;
  std::map<std::string, std::unique_ptr<PoseExtrapolatorInterface>>
      extrapolators_;
  std::vector<std::string> lidar_frames_;
  bool tf_got;
  double cur_det_time_;
  bool lidar_initialised_;
  int init_imu_cnt_;
};

#endif // PREHANDLEMSGS_H
