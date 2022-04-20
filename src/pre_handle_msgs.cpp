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

#include "pre_handle_msgs.h"
#include <eigen_conversions/eigen_msg.h>

PreHandleMsgs::PreHandleMsgs(const ros::NodeHandle& n)
    : nh_(n), tfListener_(tfBuffer_), deskewFlag(0), init_imu_cnt_(0)
{

  sub_imu_ = nh_.subscribe<sensor_msgs::Imu>(imuTopic, 2000,
                                             &PreHandleMsgs::imuHandler, this);
  sub_odom_ = nh_.subscribe<nav_msgs::Odometry>("/emma_odom", 2000,
                                                &PreHandleMsgs::odometryHandler,
                                                this); // use velocity only
  sub_pre_odom_ = nh_.subscribe<nav_msgs::Odometry>(
      odomTopic + "_incremental", 2000, &PreHandleMsgs::preOdomHandler, this);
  use_lidars_ = false;

  pubExtractedCloud = nh_.advertise<sensor_msgs::PointCloud2>("/extracted_cloud", 1);
  pubLaserCloudInfo =
      nh_.advertise<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 100);
  pubCornerPoints = nh_.advertise<sensor_msgs::PointCloud2>(
      "lio_sam/feature/cloud_corner", 1);
  pubSurfacePoints = nh_.advertise<sensor_msgs::PointCloud2>(
      "lio_sam/feature/cloud_surface", 1);
  pose_sub_ = nh_.subscribe("/lio_sam/matched_pose", 50,
                            &PreHandleMsgs::handlePose, this);
  std::vector<std::string> lidar_topics_;
  if (use_lidars_)
  {
    size_of_lidar_sensors_ = 2;
    lidar_topics_.resize(size_of_lidar_sensors_);
    lidar_topics_[0] = "/pointcloud_front";
    lidar_topics_[1] = "/pointcloud_back";
    lidar_frames_.reserve(size_of_lidar_sensors_);
    lidar_frames_.push_back("lidar_link");
    lidar_frames_.push_back("lidar_link2");
    lidar1_sub_.subscribe(nh_, lidar_topics_[0], 10);
    lidar2_sub_.subscribe(nh_, lidar_topics_[1], 10);
    lidarsSyncPolicy policy_lidars(15);
    lidarsSync_ = new message_filters::Synchronizer<lidarsSyncPolicy>(
        lidarsSyncPolicy(policy_lidars), lidar1_sub_, lidar2_sub_);
    lidarsSync_->registerCallback(
        boost::bind(&PreHandleMsgs::lidarsCallback, this, _1, _2));
  }
  else
  {
    lidar_topics_.resize(1);
    lidar_topics_[0] = pointCloudTopic;
    //     lidar_topics_[1] = "/pointcloud_back";
    subLaserCloud = nh_.subscribe<sensor_msgs::PointCloud2>(
        lidar_topics_[0], 0, &PreHandleMsgs::lidarCallback, this,
        ros::TransportHints().tcpNoDelay());
    size_of_lidar_sensors_ = 1;
    lidar_frames_.reserve(size_of_lidar_sensors_);
    lidar_frames_.push_back(lidarFrame);
    
  }

  //   lidar_frames_.push_back("lidar_link2");
  //   lidar_frames_.push_back("tf_pub");
  LOG(WARNING) << "lidar_frames_ size: " << lidar_frames_.size();
  tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
  LOG(WARNING) << "111";
  tmpHesaiCloudIn.reset(new pcl::PointCloud<HesaiPointXYZIRT>());
  LOG(WARNING) << "111";
  image_projection_.reset(new ImageProjection());
  LOG(WARNING) << "111";  
  cloud_info_.reset(new lio_sam::cloud_info());
  tf_got = false;
  LOG(WARNING) << "111";
  resetParameters();
  LOG(WARNING) << "111";
  lidar_initialised_ = false;
}

PreHandleMsgs::~PreHandleMsgs() {}

void PreHandleMsgs::resetParameters()
{
  points_poses_.clear();
  for (int i = 0; i < queueLength; ++i)
  {
    imuTime[i] = 0;
    imuRotX[i] = 0;
    imuRotY[i] = 0;
    imuRotZ[i] = 0;
  }
}

void PreHandleMsgs::lidarCallback(
    const sensor_msgs::PointCloud2::ConstPtr& lidar_msg)
{
  if (lidar_frames_.empty())
  {
    lidar_frames_.push_back(lidar_msg->header.frame_id);
  }
  if (extrapolators_[lidar_frames_[0]] == nullptr  || init_imu_cnt_ < 200)
    return;
  cloud_queue_[lidar_msg->header.frame_id].push_back(*lidar_msg);
  if(cloud_queue_[lidar_msg->header.frame_id].size() < 2)
    return;

  if (!tf_got)
  {
    geometry_msgs::TransformStamped lidar2base_stamped;
    try
    {
      lidar2base_stamped = tfBuffer_.lookupTransform(
          "base_footprint", lidar_msg->header.frame_id, ros::Time(0));
      tf_got = true;
      Eigen::Affine3d transform;
      tf::transformMsgToEigen(lidar2base_stamped.transform, transform);
      lidars2base_[lidar_msg->header.frame_id] = transform.cast<float>();
    }
    catch (tf2::TransformException& ex)
    {
      LOG_EVERY_N(INFO, 20) << ("waiting for tf_static of base2lidar !");
      return;
    }
  }
  // resort points using time
  //   points_times_.clear();

  pcl::PointCloud<PointXYZIRT>::Ptr lidar1_cloud;
  lidar1_cloud.reset(new pcl::PointCloud<PointXYZIRT>());
  sensor_msgs::PointCloud2 cur_msg =
      cloud_queue_[lidar_msg->header.frame_id].front();

  cur_time_ = 1e10;
  cur_start_time_ = cur_msg.header.stamp.toSec();
  points_poses_.reserve(cur_msg.height * cur_msg.width);
  cachePointCloud(cur_msg, lidar1_cloud, cur_time_, 0.0);
  cloud_queue_[cur_msg.header.frame_id].pop_front();

  if (!deskewInfo())
    return;
  image_projection_->cloudHandler(lidar1_cloud, points_poses_,
                                  cloud_info_->imuAvailable,
                                  cloud_info_->odomAvailable,
                                  lidars2base_[cur_msg.header.frame_id]);
  pcl::PointCloud<PointType>::Ptr corner_cloud;
  pcl::PointCloud<PointType>::Ptr surface_cloud;
  pcl::PointCloud<PointType>::Ptr cloud_deskewed;
  corner_cloud.reset(new pcl::PointCloud<PointType>());
  surface_cloud.reset(new pcl::PointCloud<PointType>());
  cloud_deskewed.reset(new pcl::PointCloud<PointType>());

  std::shared_ptr<lio_sam::cloud_info> cloud_info;
  image_projection_->getFeatureCloud(lidars2base_[cur_msg.header.frame_id],
                                     corner_cloud, surface_cloud, cloud_info);
  *cloud_deskewed += *corner_cloud;
  *cloud_deskewed += *surface_cloud;
  pcl::toROSMsg(*corner_cloud, cloud_info_->cloud_corner);
  pcl::toROSMsg(*surface_cloud, cloud_info_->cloud_surface);
  pcl::toROSMsg(*cloud_deskewed, cloud_info_->cloud_deskewed);
  cloud_info_->header.stamp = ros::Time(cur_time_);
  pubLaserCloudInfo.publish(*cloud_info_);
  
  sensor_msgs::PointCloud2 msg2;
  msg2 = cloud_info_->cloud_deskewed;
  msg2.header.frame_id = "base_footprint";
  msg2.header.stamp = cloud_info_->header.stamp;
  pubExtractedCloud.publish(msg2);
  
  resetParameters();
}

void PreHandleMsgs::lidarsCallback(
    const sensor_msgs::PointCloud2::ConstPtr& lidar1_msg,
    const sensor_msgs::PointCloud2::ConstPtr& lidar2_msg)
{
  //   if(lidar_frames_.empty())
  //   {
  //     lidar_frames_.push_back(lidar1_msg->header.frame_id);
  //     lidar_frames_.push_back(lidar2_msg->header.frame_id);
  //     lidar_frames_.push_back("tf_pub");
  //   }
  //   static int kkk = 0;
  if (extrapolators_[lidar_frames_[0]] == nullptr || init_imu_cnt_ < 200)
    return;
  cloud_queue_[lidar1_msg->header.frame_id].push_back(*lidar1_msg);
  cloud_queue_[lidar2_msg->header.frame_id].push_back(*lidar2_msg);
     if(cloud_queue_[lidar1_msg->header.frame_id].size() < 2)
       return;

  if (!tf_got)
  {
    geometry_msgs::TransformStamped lidar2base_stamped1, lidar2base_stamped2;
    try
    {
      lidar2base_stamped1 = tfBuffer_.lookupTransform(
          "base_footprint", lidar1_msg->header.frame_id, ros::Time(0));
      lidar2base_stamped2 = tfBuffer_.lookupTransform(
          "base_footprint", lidar2_msg->header.frame_id, ros::Time(0));
      tf_got = true;
      Eigen::Affine3d transform1, transform2;
      tf::transformMsgToEigen(lidar2base_stamped1.transform, transform1);
      tf::transformMsgToEigen(lidar2base_stamped2.transform, transform2);
      lidars2base_[lidar1_msg->header.frame_id] = transform1.cast<float>();
      lidars2base_[lidar2_msg->header.frame_id] = transform2.cast<float>();
    }
    catch (tf2::TransformException& ex)
    {
      LOG_EVERY_N(INFO, 20) << ("waiting for cartographer !");
      return;
    }
  }

  //   points_times_.clear();
  // resort points using time
  pcl::PointCloud<PointXYZIRT>::Ptr lidar1_cloud;
  lidar1_cloud.reset(new pcl::PointCloud<PointXYZIRT>());
  pcl::PointCloud<PointXYZIRT>::Ptr lidar2_cloud;
  lidar2_cloud.reset(new pcl::PointCloud<PointXYZIRT>());
  sensor_msgs::PointCloud2 cur_msg =
      cloud_queue_[lidar1_msg->header.frame_id].front();
  sensor_msgs::PointCloud2 cur_msg2 =
      cloud_queue_[lidar2_msg->header.frame_id].front();
  double end_time2;
  cur_time_ = 1e15;
  cur_start_time_ = cur_msg.header.stamp.toSec();
  if (cur_msg2.header.stamp.toSec() < cur_start_time_)
    cur_start_time_ = cur_msg2.header.stamp.toSec();

  points_poses_.reserve(cur_msg.height * cur_msg.width +
                        cur_msg2.height * cur_msg2.width);
  if(cur_msg.header.stamp.toSec() == cur_start_time_)
  {
    cachePointCloud(cur_msg, lidar1_cloud, cur_time_,0.0);
    cachePointCloud(cur_msg2, lidar2_cloud, end_time2,
                    cur_msg2.header.stamp.toSec() - cur_start_time_);
  }
  else
  {
    cachePointCloud(cur_msg2, lidar2_cloud, cur_time_,0.0);
    cachePointCloud(cur_msg, lidar1_cloud, end_time2,
                    cur_msg.header.stamp.toSec() - cur_start_time_);
  }
//   if (end_time2 < cur_time_)
//     cur_time_ = end_time2;
  cloud_queue_[cur_msg.header.frame_id].pop_front();
  cloud_queue_[cur_msg2.header.frame_id].pop_front();
  if (!deskewInfo())
    return;

  pcl::PointCloud<PointType>::Ptr corner_cloud;
  pcl::PointCloud<PointType>::Ptr surface_cloud;
  pcl::PointCloud<PointType>::Ptr cloud_deskewed;
  corner_cloud.reset(new pcl::PointCloud<PointType>());
  surface_cloud.reset(new pcl::PointCloud<PointType>());
  cloud_deskewed.reset(new pcl::PointCloud<PointType>());
  std::shared_ptr<lio_sam::cloud_info> cloud_info;

  image_projection_->cloudHandler(lidar1_cloud, points_poses_,
                                  cloud_info_->imuAvailable,
                                  cloud_info_->odomAvailable, lidars2base_[cur_msg.header.frame_id]);
  image_projection_->getFeatureCloud(lidars2base_[cur_msg.header.frame_id],
                                     corner_cloud, surface_cloud, cloud_info);
  image_projection_->cloudHandler(lidar2_cloud, points_poses_,
                                  cloud_info_->imuAvailable,
                                  cloud_info_->odomAvailable, lidars2base_[cur_msg2.header.frame_id]);
  image_projection_->getFeatureCloud(lidars2base_[cur_msg2.header.frame_id],
                                     corner_cloud, surface_cloud, cloud_info);
  *cloud_deskewed += *corner_cloud;
  *cloud_deskewed += *surface_cloud;

  pcl::toROSMsg(*corner_cloud, cloud_info_->cloud_corner);
  pcl::toROSMsg(*surface_cloud, cloud_info_->cloud_surface);
  pcl::toROSMsg(*cloud_deskewed, cloud_info_->cloud_deskewed);
  cloud_info_->header.stamp = ros::Time(cur_time_);
  pubLaserCloudInfo.publish(*cloud_info_);
  //   sensor_msgs::PointCloud2 msg2;
  //   msg2 = cloud_info_->cloud_deskewed;
  //   msg2.header.frame_id = "base_footprint";
  //   msg2.header.stamp = cloud_info_->header.stamp;
  //   pubExtractedCloud.publish(msg2);
  resetParameters();
}

void PreHandleMsgs::imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
{
  std::lock_guard<std::mutex> lock2(imu_lock_);
  if (lidar_frames_.empty())
    return;
  if (!initialised_)
  {
    initialised_ = true;
    imu_tracker_.reset(new ImuTracker(10.0, imuMsg->header.stamp.toSec()));
  }
  if(init_imu_cnt_ <= 200)
    init_imu_cnt_++;
  sensor_msgs::Imu thisImu = ParamServer::imuConverter(*imuMsg, imu_tracker_);
  imuQueue.push_back(thisImu);
  auto imu_data = ToImuData(imuMsg);
  if (extrapolators_[lidar_frames_[0]] == nullptr)
  {
    for (unsigned int i = 0; i < lidar_frames_.size(); i++)
      extrapolators_[lidar_frames_[i]] =
          PoseExtrapolatorInterface::CreateWithImuData({imu_data}, {});
  }
  else
    for (auto& extrapolator : extrapolators_)
      extrapolator.second->AddImuData(imu_data);
  //     pose_extrapolator_->AddImuData(imu_data);
  if (0)
  {
    if (extrapolators_[lidar_frames_[0]]->GetLastPoseTime() ==
        extrapolators_[lidar_frames_[0]]->time_min())
      return;
    Rigid3d temp_pose =
        extrapolators_["tf_pub"]->ExtrapolatePose(imuMsg->header.stamp.toSec());
    static tf::TransformBroadcaster tfOdom2BaseLink;
    geometry_msgs::Pose tf_pose;
    tf_pose.position.x = temp_pose.translation().x();
    tf_pose.position.y = temp_pose.translation().y();
    tf_pose.position.z = temp_pose.translation().z();

    tf_pose.orientation.x = temp_pose.rotation().x();
    tf_pose.orientation.y = temp_pose.rotation().y();
    tf_pose.orientation.z = temp_pose.rotation().z();
    tf_pose.orientation.w = temp_pose.rotation().w();
    tf::Transform tCur;
    tf::poseMsgToTF(tf_pose, tCur);
    tf::StampedTransform odom_2_baselink = tf::StampedTransform(
        tCur, imuMsg->header.stamp, odometryFrame, baselinkFrame);
    tfOdom2BaseLink.sendTransform(odom_2_baselink);
  }
}

void PreHandleMsgs::odometryHandler(
    const nav_msgs::Odometry::ConstPtr& odometryMsg)
{
  std::lock_guard<std::mutex> lock2(odoLock);
  if (lidar_frames_.empty())
    return;
  for (auto& extrapolator : extrapolators_)
    extrapolator.second->AddOdometryData(ToOdometryData(odometryMsg));
}

void PreHandleMsgs::preOdomHandler(
    const nav_msgs::Odometry::ConstPtr& odometryMsg)
{
  // FIXME1: useless now
  std::lock_guard<std::mutex> lock2(odoLock);
  odomQueue.push_back(*odometryMsg);
}

void PreHandleMsgs::cachePointCloud(
    sensor_msgs::PointCloud2& currentCloudMsg,
    pcl::PointCloud<PointXYZIRT>::Ptr lidar_cloud, double& point_time_end,
    const double& det_time)
{
  std::set<float> points_times;
  if (sensor == SensorType::VELODYNE)
  {
    pcl::PointCloud<PointXYZIRT>::Ptr tmp_lidar_cloud;
    pcl::moveFromROSMsg(currentCloudMsg, *tmp_lidar_cloud);
    lidar_cloud->reserve(tmp_lidar_cloud->size());
    
//     LOG(WARNING) << lidar_cloud->points[0].time <<", " <<  lidar_cloud->points.back().time ;
    for (auto& point : lidar_cloud->points)
    {
      int rowIdn = point.ring;
      if (rowIdn < 0 || rowIdn >= N_SCAN)
        continue;

      if (rowIdn % downsampleRate != 0)
        continue;

      point.time += det_time;
      if (fabs(point.time) > 1.0)
        point.time = 0;
      points_times.insert(point.time);
      lidar_cloud->push_back(point);
    }
  }
  else if (sensor == SensorType::OUSTER)
  {
    // Convert to Velodyne format
    pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
    lidar_cloud->points.reserve(tmpOusterCloudIn->size());
    lidar_cloud->is_dense = tmpOusterCloudIn->is_dense;
    for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
    {
      auto& src = tmpOusterCloudIn->points[i];
      int rowIdn = src.ring;
      if (rowIdn < 0 || rowIdn >= N_SCAN)
        continue;

      if (rowIdn % downsampleRate != 0)
        continue;
      
      PointXYZIRT dst;
      dst.x = src.x;
      dst.y = src.y;
      dst.z = src.z;
      dst.intensity = src.intensity;
      dst.ring = src.ring;
      dst.time = src.t * 1e-9f + det_time;
      points_times.insert(dst.time);
      lidar_cloud->push_back(dst);
    }
  }
  else if (sensor == SensorType::HESAI)
  {
    // Convert to Velodyne format
    pcl::moveFromROSMsg(currentCloudMsg, *tmpHesaiCloudIn);
    lidar_cloud->points.reserve(tmpHesaiCloudIn->size());
    lidar_cloud->is_dense = tmpHesaiCloudIn->is_dense;
    
    for (size_t i = 0; i < tmpHesaiCloudIn->size(); i++)
    {
      auto& src = tmpHesaiCloudIn->points[i];
      int rowIdn = src.ring;
      if (rowIdn < 0 || rowIdn >= N_SCAN)
        continue;

      if (rowIdn % downsampleRate != 0)
        continue;
      
      PointXYZIRT dst;
      dst.x = src.x;
      dst.y = src.y;
      dst.z = src.z;
      dst.intensity = src.intensity;
      dst.ring = src.ring;
      float det_cur_time = tmpHesaiCloudIn->points[i].timestamp -
                           tmpHesaiCloudIn->points.front().timestamp;

      dst.time = det_cur_time + det_time;
      points_times.insert(dst.time);
      lidar_cloud->push_back(dst);
    }
  }
  else
  {
    ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
    ros::shutdown();
  }
  lidar_cloud->resize(lidar_cloud->size());
  point_time_end = cur_start_time_ + det_time + lidar_cloud->points.back().time;
  if (det_time == 0)
    cur_det_time_ = *points_times.rbegin();
  // TODO: Will it be better to use "odometry/imu"?(make pose_extrator bias
  // considered.)
  predictPointCloudPoses(currentCloudMsg.header.frame_id, points_times);
  // check dense flag
  if (lidar_cloud->is_dense == false)
  {
    ROS_ERROR(
        "Point cloud is not in dense format, please remove NaN points first!");
    ros::shutdown();
  }

  // check ring channel
  static int ringFlag = 0;
  if (ringFlag == 0)
  {
    ringFlag = -1;
    for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
    {
      if (currentCloudMsg.fields[i].name == "ring")
      {
        ringFlag = 1;
        break;
      }
    }
    if (ringFlag == -1)
    {
      ROS_ERROR("Point cloud ring channel not available, please configure your "
                "point cloud data!");
      ros::shutdown();
    }
  }
  // check point time
  if (deskewFlag == 0)
  {
    deskewFlag = -1;
    for (auto& field : currentCloudMsg.fields)
    {
      if (field.name == "time" || field.name == "t")
      {
        deskewFlag = 1;
        break;
      }
    }
    if (deskewFlag == -1)
      ROS_WARN("Point cloud timestamp not available, deskew function disabled, "
               "system will drift significantly!");
  }
}

void PreHandleMsgs::predictPointCloudPoses(const string& frame_id,
                                           const set<float>& points_times)
{
  int i =0;
  
  for (const auto& cur_time : points_times)
  {
    double time_point = cur_start_time_ + cur_time;
    if (time_point < extrapolators_[frame_id]->GetLastExtrapolatedTime())
    {
      time_point = extrapolators_[frame_id]->GetLastExtrapolatedTime();
    }
    points_poses_[cur_time] = extrapolators_[frame_id]
                                  ->ExtrapolatePose(time_point)
                                  .cast<float>()
                                  .ToAffine();
    // if(i == 0 || i == points_times.size() -2)
    // {
    //   LOG(WARNING) <<cur_time << ", " <<points_poses_[cur_time].matrix();
    // }
    i++;
  }
  
//   float max_det_time = *(points_times.rbegin());
//   for(const auto& cur_time : points_times)
//   {
//     points_poses_[cur_time] = points_poses_[cur_det_time_].inverse() * points_poses_[cur_time];
//   }
  
}

bool PreHandleMsgs::deskewInfo()
{
  {
    Eigen::Affine3f cur_pose = points_poses_[cur_det_time_];
    cloud_info_->initialGuessX = cur_pose.translation()[0];
    cloud_info_->initialGuessY = cur_pose.translation()[1];
    cloud_info_->initialGuessZ = cur_pose.translation()[2];
    Eigen::Quaternionf q_temp = (Eigen::Quaternionf)cur_pose.linear();
    Eigen::Vector3f eul = q_temp.toRotationMatrix().eulerAngles(2, 1, 0);
    cloud_info_->initialGuessRoll = eul(2);
    cloud_info_->initialGuessPitch = eul(1);
    cloud_info_->initialGuessYaw = eul(0);

//     cloud_info_->odomAvailable = true;
    for (auto& it : points_poses_)
      it.second = cur_pose.inverse() * it.second;
  }
  double roll = 0.0, pitch = 0.0, yaw = 0.0;
  if (!lidar_initialised_)
  {
    Eigen::Quaterniond gravity_align =
        extrapolators_[lidar_frames_[0]]->EstimateGravityOrientation(cur_time_);
    LOG(WARNING) <<"gravity_align: " <<  gravity_align.coeffs();
    if (gravity_align.w() == 0 && gravity_align.x() == 0 &&
        gravity_align.y() == 0 && gravity_align.z() == 0)
      return false;
    
    tf::Matrix3x3(tf::Quaternion(gravity_align.x(), gravity_align.y(),
                                 gravity_align.z(), gravity_align.w()))
        .getRPY(roll, pitch, yaw);
    cloud_info_->initialGuessRoll = roll;
    cloud_info_->initialGuessYaw = yaw;
    cloud_info_->initialGuessPitch = pitch;
    lidar_initialised_ = true;
  }
  //   odomDeskewInfo();
  imuDeskewInfo();
  Eigen::Quaterniond gravity_align =
        extrapolators_[lidar_frames_[0]]->EstimateGravityOrientation(cur_time_);
  tf::Matrix3x3(tf::Quaternion(gravity_align.x(), gravity_align.y(),
                            gravity_align.z(), gravity_align.w()))
              .getRPY(roll, pitch, yaw);
  cloud_info_->imuRollInit = roll;
  cloud_info_->imuPitchInit = pitch;
  cloud_info_->imuYawInit = yaw;
  return true;
}

void PreHandleMsgs::imuDeskewInfo()
{
  //   double timeScanCur = c;
  //   double timeScanEnd = cur_start_time_ + *points_times_.rbegin();
  //   LOG(WARNING) << timeScanCur << ", " << timeScanEnd;
  cloud_info_->imuAvailable = false;

  while (!imuQueue.empty())
  {
    if (imuQueue.front().header.stamp.toSec() < cur_time_ - 0.02)
      imuQueue.pop_front();
    else
      break;
  }

  if (imuQueue.empty())
    return;

  imuPointerCur = 0;

//   for (int i = 0; i < (int)imuQueue.size(); ++i)
//   {
//     sensor_msgs::Imu thisImuMsg = imuQueue[i];
//     double currentImuTime = thisImuMsg.header.stamp.toSec();

    // get roll, pitch, and yaw estimation for this scan
//     if (currentImuTime <= cur_time_)
//     {
//       // orientation is estimated by imu_tracker
// //       Eigen::Quaterniond tmp =
// //         extrapolators_[lidar_frames_[0]]->EstimateGravityOrientation(currentImuTime);
// //       thisImuMsg.orientation.x = tmp.x();
// //       thisImuMsg.orientation.y = tmp.y();
// //       thisImuMsg.orientation.z = tmp.z();
// //       thisImuMsg.orientation.w = tmp.w();
//       
//       imuRPY2rosRPY(&thisImuMsg, &cloud_info_->imuRollInit,
//                     &cloud_info_->imuPitchInit, &cloud_info_->imuYawInit);
//     }
//     else
//       break;
//     //     if (currentImuTime > timeScanEnd + 0.01)
//     //       break;
// 
//     //     if (imuPointerCur == 0){
//     //       imuRotX[0] = 0;
//     //       imuRotY[0] = 0;
//     //       imuRotZ[0] = 0;
//     //       imuTime[0] = currentImuTime;
//     //       ++imuPointerCur;
//     //       continue;
//     //     }
//     //
//     //     // get angular velocity
//     //     double angular_x, angular_y, angular_z;
//     //     imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y,
//     //     &angular_z);
//     //
//     //     // integrate rotation
//     //     double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
//     //     imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x *
//     //     timeDiff;
//     //     imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y *
//     //     timeDiff;
//     //     imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z *
//     //     timeDiff;
//     //     imuTime[imuPointerCur] = currentImuTime;
//     //     ++imuPointerCur;
//   }
  //LOG(WARNING) <<"roll and pitch: " << cloud_info_->imuRollInit <<", " << cloud_info_->imuPitchInit ;
  //   --imuPointerCur;
  //
  //   if (imuPointerCur <= 0)
  //     return;
  
//   {
//     static tf::TransformBroadcaster tfOdom2BaseLink;
//     geometry_msgs::Pose tf_pose;
//     tf_pose.position.x = 0;
//     tf_pose.position.y = 0;
//     tf_pose.position.z = 0;
//     sensor_msgs::Imu cur_imu_msg = imuQueue.front();
//     tf_pose.orientation.x = cur_imu_msg.orientation.x;
//     tf_pose.orientation.y = cur_imu_msg.orientation.y;
//     tf_pose.orientation.z = cur_imu_msg.orientation.z;
//     tf_pose.orientation.w = cur_imu_msg.orientation.w;
//     tf::Transform tCur;
//     tf::poseMsgToTF(tf_pose, tCur);
//     tf::StampedTransform odom_2_baselink = tf::StampedTransform(
//         tCur, ros::Time::now(), odometryFrame, "gravity_align2");
//     tfOdom2BaseLink.sendTransform(odom_2_baselink);
//   }
  cloud_info_->imuAvailable = true;
}

void PreHandleMsgs::odomDeskewInfo()
{

  cloud_info_->odomAvailable = false;
  nav_msgs::Odometry lastest_msg;
  while (!odomQueue.empty())
  {
    if (odomQueue.front().header.stamp.toSec() < cur_time_)
    {
      lastest_msg = odomQueue.front();
      odomQueue.pop_front();
    }
    else
      break;
  }
  if (odomQueue.empty())
  {
    return;
  }

  if (odomQueue.front().header.stamp.toSec() > cur_time_)
    odomQueue.push_front(lastest_msg);

  if (lastest_msg.header.stamp.toSec() > cur_time_)
    return;

  // get start odometry at the beinning of the scan
  nav_msgs::Odometry startOdomMsg;

  for (int i = 0; i < (int)odomQueue.size(); ++i)
  {
    startOdomMsg = odomQueue[i];

    if (ROS_TIME(&startOdomMsg) < cur_time_)
      continue;
    else
      break;
  }

  tf::Quaternion orientation;
  tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

  double roll, pitch, yaw;
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  // Initial guess used in mapOptimization
  cloud_info_->initialGuessX = startOdomMsg.pose.pose.position.x;
  cloud_info_->initialGuessY = startOdomMsg.pose.pose.position.y;
  cloud_info_->initialGuessZ = startOdomMsg.pose.pose.position.z;
  cloud_info_->initialGuessRoll = roll;
  cloud_info_->initialGuessPitch = pitch;
  cloud_info_->initialGuessYaw = yaw;

  cloud_info_->odomAvailable = true;
  // get end odometry at the end of the scan
}

void PreHandleMsgs::findPosition(double relTime, float* posXCur, float* posYCur,
                                 float* posZCur)
{
  *posXCur = 0;
  *posYCur = 0;
  *posZCur = 0;

  // If the sensor moves relatively slow, like walking speed, positional deskew
  // seems to have little benefits. Thus code below is commented.

  // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
  //     return;

  // float ratio = relTime / (timeScanEnd - timeScanCur);

  // *posXCur = ratio * odomIncreX;
  // *posYCur = ratio * odomIncreY;
  // *posZCur = ratio * odomIncreZ;
}

void PreHandleMsgs::findRotation(double pointTime, float* rotXCur,
                                 float* rotYCur, float* rotZCur)
{
  *rotXCur = 0;
  *rotYCur = 0;
  *rotZCur = 0;

  int imuPointerFront = 0;
  while (imuPointerFront < imuPointerCur)
  {
    if (pointTime < imuTime[imuPointerFront])
      break;
    ++imuPointerFront;
  }
  float roll, pitch, yaw;
  if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
  {
    roll = imuRotX[imuPointerFront];
    pitch = imuRotY[imuPointerFront];
    yaw = imuRotZ[imuPointerFront];
  }
  else
  {
    int imuPointerBack = imuPointerFront - 1;
    double ratioFront = (pointTime - imuTime[imuPointerBack]) /
                        (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
    double ratioBack = (imuTime[imuPointerFront] - pointTime) /
                       (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

    roll = imuRotX[imuPointerFront] * ratioFront +
           imuRotX[imuPointerBack] * ratioBack;
    pitch = imuRotY[imuPointerFront] * ratioFront +
            imuRotY[imuPointerBack] * ratioBack;
    yaw = imuRotZ[imuPointerFront] * ratioFront +
          imuRotZ[imuPointerBack] * ratioBack;
  }
  //   Eigen::Matrix3d rot_in_lidar =
  //   lidar2base_q_.conjugate().toRotationMatrix() *
  //   eulToMatrix(Eigen::Vector3d(yaw, pitch, roll));
  Eigen::Quaterniond q_in_base(eulToMatrix(Eigen::Vector3d(yaw, pitch, roll)));
  double new_roll, new_pitch, new_yaw;
  tf::Matrix3x3(tf::Quaternion(q_in_base.x(), q_in_base.y(), q_in_base.z(),
                               q_in_base.w()))
      .getRPY(new_roll, new_pitch, new_yaw);
  *rotXCur = new_roll;
  *rotYCur = new_pitch;
  *rotZCur = new_yaw;
}

void PreHandleMsgs::handlePose(const geometry_msgs::PoseStampedConstPtr& msg)
{
  // add pose from imuPreintegration to ensure initial guess gravity align
  //   LOG(INFO) << ToRigid3d(msg->pose) <<endl;
//   LOG(INFO) << "add pose: " << ToRigid3d(msg->pose) ;
  for (auto& extrapolator : extrapolators_)
    extrapolator.second->AddPose(msg->header.stamp.toSec(),
                                 ToRigid3d(msg->pose));
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_INFO);
  FLAGS_colorlogtostderr = true;
  ros::init(argc, argv, "pre_handle_msgs_node");
  ros::NodeHandle n;
  PreHandleMsgs p(n);
  ros::spin();
  return 1;
}
