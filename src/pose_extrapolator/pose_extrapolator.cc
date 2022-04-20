/*
 * Copyright 2017 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "transform.h"
#include "glog/logging.h"
#include "ros/ros.h"

PoseExtrapolator::PoseExtrapolator(const double& pose_queue_duration,
                                   const double& imu_gravity_time_constant)
    : pose_queue_duration_(pose_queue_duration),
      gravity_time_constant_(imu_gravity_time_constant),
      cached_extrapolated_pose_{time_min(), Rigid3d::Identity()}
{
}

std::unique_ptr<PoseExtrapolator>
PoseExtrapolator::InitializeWithImu(const double pose_queue_duration,
                                    const double imu_gravity_time_constant,
                                    const sensor::ImuData& imu_data)
{
  auto extrapolator = absl::make_unique<PoseExtrapolator>(
      pose_queue_duration, imu_gravity_time_constant);
  extrapolator->AddImuData(imu_data);
  extrapolator->imu_tracker_ =
      absl::make_unique<ImuTracker>(imu_gravity_time_constant, imu_data.time);
  extrapolator->imu_tracker_->AddImuLinearAccelerationObservation(
      imu_data.linear_acceleration);
  extrapolator->imu_tracker_->AddImuAngularVelocityObservation(
      imu_data.angular_velocity);
  extrapolator->imu_tracker_->Advance(imu_data.time);
  extrapolator->AddPose(
      imu_data.time,
      Rigid3d::Rotation(extrapolator->imu_tracker_->orientation()));
  return extrapolator;
}

double PoseExtrapolator::GetLastPoseTime() const
{
  if (timed_pose_queue_.empty())
  {
    return time_min();
  }
  return timed_pose_queue_.back().time;
}

double PoseExtrapolator::GetLastExtrapolatedTime() const
{
  if (!extrapolation_imu_tracker_)
  {
    return time_min();
  }
  return extrapolation_imu_tracker_->time();
}

Rigid3d PoseExtrapolator::GetLastPose() const
{
  return timed_pose_queue_.back().pose;
}

void PoseExtrapolator::AddPose(const double time, const Rigid3d& pose)
{
  if (imu_tracker_ == nullptr)
  {
    double tracker_start = time;
    if (!imu_data_.empty())
    {
      tracker_start = std::min(tracker_start, imu_data_.front().time);
    }
    imu_tracker_ =
        absl::make_unique<ImuTracker>(gravity_time_constant_, tracker_start);
  }
  timed_pose_queue_.push_back(TimedPose{time, pose});
  while (timed_pose_queue_.size() > 2 &&
         timed_pose_queue_[1].time <= time - pose_queue_duration_)
  {
    timed_pose_queue_.pop_front();
  }
  UpdateVelocitiesFromPoses();
  AdvanceImuTracker(time, imu_tracker_.get());
  TrimImuData();
  TrimOdometryData();
  odometry_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
  extrapolation_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
}

void PoseExtrapolator::AddImuData(const sensor::ImuData& imu_data)
{
  if ((!timed_pose_queue_.empty()) &&
      imu_data.time < timed_pose_queue_.back().time)
  {
    LOG(WARNING) << "imu: " << (imu_data.time) << " < "
                 << (timed_pose_queue_.back().time);
    return;
  }

  imu_data_.push_back(imu_data);
  TrimImuData();
  Eigen::Vector3d imu_ang_vel = imu_data_.front().angular_velocity;
  Eigen::Vector3d err_ang = angular_velocity_from_poses_ - imu_ang_vel;

  if (fabs(err_ang(2)) > 0.8 && imu_ang_vel(2) != 0. && 0)
    LOG(ERROR) << "imu angular velocity is unusual!\n"
               << "Time is  " << (imu_data_.front().time) << "\n"
               << "oldest_timed_pose.time: " << (timed_pose_queue_.front().time)
               << "\n"
               << "newest_timed_pose.time: " << (timed_pose_queue_.back().time)
               << "\n"
               << "The angular_velocity_from_poses_ is : "
               << angular_velocity_from_poses_.transpose() << "\n"
               << "The imu_ang_vel is :  " << imu_ang_vel.transpose();
}

void PoseExtrapolator::AddOdometryData(
    const sensor::OdometryData& odometry_data)
{
  if ((!timed_pose_queue_.empty()) &&
      odometry_data.time < timed_pose_queue_.back().time)
  {
    LOG(WARNING) << "odom: " << (odometry_data.time) << " < "
                 << (timed_pose_queue_.back().time);
    return;
  }

  odometry_data_.push_back(odometry_data);
  TrimOdometryData();
  if (odometry_data_.size() < 2)
  {
    return;
  }
  // TODO(whess): Improve by using more than just the last two odometry poses.
  // Compute extrapolation in the tracking frame.
  const sensor::OdometryData& odometry_data_oldest = odometry_data_.front();
  const sensor::OdometryData& odometry_data_newest = odometry_data_.back();

  // const double odometry_time_delta =
  //     (odometry_data_oldest.time - odometry_data_newest.time);
  // const Rigid3d odometry_pose_delta =
  //     odometry_data_newest.pose.inverse() * odometry_data_oldest.pose;
  angular_velocity_from_odometry_ = odometry_data_newest.twist_angular;
  // transform::RotationQuaternionToAngleAxisVector(
  //     odometry_pose_delta.rotation()) /
  // odometry_time_delta;
  if (timed_pose_queue_.empty())
  {
    return;
  }
  const Eigen::Vector3d
      linear_velocity_in_tracking_frame_at_newest_odometry_time =
          odometry_data_newest.twist_linear;
  // odometry_pose_delta.translation() / odometry_time_delta;
  // std::cout << "odom vel: "
  //           << (odometry_pose_delta.translation() / odometry_time_delta
  //           ).transpose()
  //           << std::endl;
  // std::cout << "\033[32m  pose vel:" <<
  // linear_velocity_from_poses_.transpose() << std::endl;
  // std::cout << "odom twist vel: " <<
  // linear_velocity_from_odometry_.transpose()<<"\033[0m"<<std::endl;
  const Eigen::Quaterniond orientation_at_newest_odometry_time =
      timed_pose_queue_.back().pose.rotation() *
      ExtrapolateRotation(odometry_data_newest.time,
                          odometry_imu_tracker_.get());
  linear_velocity_from_odometry_ =
      orientation_at_newest_odometry_time *
      linear_velocity_in_tracking_frame_at_newest_odometry_time;

  Eigen::Vector3d err =
      angular_velocity_from_poses_ - angular_velocity_from_odometry_;

  if (fabs(err(2)) > 0.8 && angular_velocity_from_odometry_(2) != 0. && 0)
    LOG(ERROR) << "odom angular velocity is unusual! \n"
               << "Time is  " << (odometry_data.time) << "\n"
               << "oldest_timed_pose.time: " << (timed_pose_queue_.front().time)
               << "\n"
               << "newest_timed_pose.time: " << (timed_pose_queue_.back().time)
               << "\n"
               << "The angular_velocity_from_poses_ is : "
               << angular_velocity_from_poses_.transpose() << "\n"
               << "The angular_velocity_from_odometry_ is :  "
               << angular_velocity_from_odometry_.transpose();

  Eigen::Vector3d err_linear =
      linear_velocity_from_poses_ - linear_velocity_from_odometry_;
  if ((fabs(err_linear(0)) > 2. || fabs(err_linear(1)) > 2.) &&
      (linear_velocity_from_odometry_(0) != 0. ||
       linear_velocity_from_odometry_(1) != 0.) &&
      0)
    LOG(ERROR) << "linear velocity is unusual ! \n"
               << "Time is  " << (odometry_data.time) << "\n"
               << "oldest_timed_pose.time: " << (timed_pose_queue_.front().time)
               << "\n"
               << "newest_timed_pose.time: " << (timed_pose_queue_.back().time)
               << "\n"
               << "The linear_velocity_from_poses_ is : "
               << linear_velocity_from_poses_.transpose() << "\n"
               << "The linear_velocity_from_odometry_ is :  "
               << linear_velocity_from_odometry_.transpose();
}

Rigid3d PoseExtrapolator::ExtrapolatePose(const double time)
{

  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  if (cached_extrapolated_pose_.time != time)
  {
    const Eigen::Vector3d translation =
        ExtrapolateTranslation(time) + newest_timed_pose.pose.translation();
    const Eigen::Quaterniond rotation =
        newest_timed_pose.pose.rotation() *
        ExtrapolateRotation(time, extrapolation_imu_tracker_.get());
    cached_extrapolated_pose_ = TimedPose{time, Rigid3d{translation, rotation}};
  }
  return cached_extrapolated_pose_.pose;
}

Eigen::Quaterniond
PoseExtrapolator::EstimateGravityOrientation(const double time)
{
  if (imu_tracker_->time() > time)
    return Eigen::Quaterniond(0, 0, 0, 0);
  ImuTracker imu_tracker = *imu_tracker_;
  AdvanceImuTracker(time, &imu_tracker);
  return imu_tracker.orientation();
}

void PoseExtrapolator::UpdateVelocitiesFromPoses()
{
  if (timed_pose_queue_.size() < 2)
  {
    // We need two poses to estimate velocities.
    return;
  }
  CHECK(!timed_pose_queue_.empty());
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const auto newest_time = newest_timed_pose.time;
  const TimedPose& oldest_timed_pose = timed_pose_queue_.front();
  const auto oldest_time = oldest_timed_pose.time;
  const double queue_delta = (newest_time - oldest_time);
  if (queue_delta < (pose_queue_duration_))
  { // 1 ms
    LOG(WARNING) << "Queue too short for velocity estimation. Queue duration: "
                 << queue_delta << " s";
    return;
  }
  const Rigid3d& newest_pose = newest_timed_pose.pose;
  const Rigid3d& oldest_pose = oldest_timed_pose.pose;
  linear_velocity_from_poses_ =
      (newest_pose.translation() - oldest_pose.translation()) / queue_delta;
  angular_velocity_from_poses_ =
      RotationQuaternionToAngleAxisVector(oldest_pose.rotation().inverse() *
                                          newest_pose.rotation()) /
      queue_delta;
}

void PoseExtrapolator::TrimImuData()
{
  while (imu_data_.size() > 1 && !timed_pose_queue_.empty() &&
         imu_data_[1].time <= timed_pose_queue_.back().time)
  {
    imu_data_.pop_front();
  }
}

void PoseExtrapolator::TrimOdometryData()
{
  while (odometry_data_.size() > 2 && !timed_pose_queue_.empty() &&
         odometry_data_[1].time <= timed_pose_queue_.back().time)
  {
    odometry_data_.pop_front();
  }
}

void PoseExtrapolator::AdvanceImuTracker(const double time,
                                         ImuTracker* const imu_tracker) const
{
  CHECK_GE(time, imu_tracker->time());
  if (imu_data_.empty() || time < imu_data_.front().time)
  {
    // There is no IMU data until 'time', so we advance the ImuTracker and use
    // the angular velocities from poses and fake gravity to help 2D stability.
    imu_tracker->Advance(time);
    imu_tracker->AddImuLinearAccelerationObservation(Eigen::Vector3d::UnitZ());
    imu_tracker->AddImuAngularVelocityObservation(
        odometry_data_.size() < 2 ? angular_velocity_from_poses_
                                  : angular_velocity_from_odometry_);
    return;
  }
  if (imu_tracker->time() < imu_data_.front().time)
  {
    // Advance to the beginning of 'imu_data_'.
    imu_tracker->Advance(imu_data_.front().time);
  }
  auto it =
      std::lower_bound(imu_data_.begin(), imu_data_.end(), imu_tracker->time(),
                       [](const sensor::ImuData& imu_data, const double& time)
                       {
                         return imu_data.time < time;
                       });
  while (it != imu_data_.end() && it->time < time)
  {
    //    LOG_EVERY_N(WARNING, 100) << "imu_linear_acceleration:" <<
    //    it->linear_acceleration[0] << '\t'
    //                               << it->linear_acceleration[1] << '\t' <<
    //                               it->linear_acceleration[2];
    imu_tracker->Advance(it->time);
    imu_tracker->AddImuLinearAccelerationObservation(it->linear_acceleration);
    imu_tracker->AddImuAngularVelocityObservation(it->angular_velocity);
    ++it;
  }
  imu_tracker->Advance(time);

  //  LOG(WARNING) << imu_tracker->orientation().inverse().toRotationMatrix() *
  //                      (-Eigen::Vector3d::UnitZ());
  //  LOG(INFO) << (imu_data_.back().linear_acceleration).normalized();
}

Eigen::Quaterniond
PoseExtrapolator::ExtrapolateRotation(const double time,
                                      ImuTracker* const imu_tracker) const
{
  CHECK_GE(time, imu_tracker->time()) << std::setprecision(16) << time << ", "
                                      << imu_tracker->time();
  AdvanceImuTracker(time, imu_tracker);
  const Eigen::Quaterniond last_orientation = imu_tracker_->orientation();

  return last_orientation.inverse() * imu_tracker->orientation();
}

Eigen::Vector3d PoseExtrapolator::ExtrapolateTranslation(double time)
{
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const double extrapolation_delta = (time - newest_timed_pose.time);

  if (odometry_data_.size() < 2)
  {
    return extrapolation_delta * linear_velocity_from_poses_;
  }
  return extrapolation_delta * linear_velocity_from_odometry_;
}

PoseExtrapolator::ExtrapolationResult
PoseExtrapolator::ExtrapolatePosesWithGravity(const std::vector<double>& times)
{
  std::vector<Rigid3f> poses;
  //   LOG(WARNING) <<std::setprecision(16) << timed_pose_queue_.back().time <<
  //   ", " << *times.begin();
  for (auto it = times.begin(); it != std::prev(times.end()); ++it)
  {
    //     poses.push_back(ExtrapolatePose(*it).cast<float>());
    poses.push_back(Rigid3f::Identity());
  }

  const Eigen::Vector3d current_velocity = odometry_data_.size() < 2
                                               ? linear_velocity_from_poses_
                                               : linear_velocity_from_odometry_;
  return ExtrapolationResult{poses, Rigid3d::Identity(), current_velocity,
                             EstimateGravityOrientation(times.back())};
}
