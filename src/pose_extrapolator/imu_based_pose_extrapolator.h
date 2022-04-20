/*
* Copyright 2018 The Cartographer Authors
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

#ifndef CARTOGRAPHER_MAPPING_IMU_BASED_POSE_EXTRAPOLATOR_H_
#define CARTOGRAPHER_MAPPING_IMU_BASED_POSE_EXTRAPOLATOR_H_

#include <deque>
#include <memory>
#include <vector>
#include "pose_extrapolator_interface.h"
#include <ceres/ceres.h>
#include "absl/container/flat_hash_map.h"
// #include "cartographer/common/ceres_solver_options.h"
// #include "cartographer/common/histogram.h"
// #include "cartographer/mapping/pose_extrapolator_interface.h"
// #include "cartographer/sensor/imu_data.h"
// #include "cartographer/transform/timestamped_transform.h"

// Uses the linear acceleration and rotational velocities to estimate a pose.

// singleton design pattern
class ImuBaseOptions
{
public:
  static ImuBaseOptions& options()
  {
    static ImuBaseOptions options;
    return options;
  }

  const double& pose_queue_duration() const { return pose_queue_duration_; }
  const double& gravity_constant() const { return gravity_constant_; }
  const double& pose_translation_weight() const
  {
    return pose_translation_weight_;
  }
  const double& pose_rotation_weight() const { return pose_rotation_weight_; }
  const double& imu_acceleration_weight() const
  {
    return imu_acceleration_weight_;
  }
  const double& imu_rotation_weight() const { return imu_rotation_weight_; }
  const double& odometry_translation_weight() const
  {
    return odometry_translation_weight_;
  }
  const double& odometry_rotation_weight() const
  {
    return odometry_rotation_weight_;
  }
  const ceres::Solver::Options& ceres_options() const
  {
    return ceres_options_;
  };

private:
  ImuBaseOptions();
  ImuBaseOptions(const ImuBaseOptions& a) = delete;
  ImuBaseOptions(ImuBaseOptions&& a) = delete;
  ImuBaseOptions& operator=(const ImuBaseOptions& a) = delete;
  ImuBaseOptions& operator=(ImuBaseOptions&& other) = delete;
  double pose_queue_duration_;
  double gravity_constant_;
  double pose_translation_weight_;
  double pose_rotation_weight_;
  double imu_acceleration_weight_;
  double imu_rotation_weight_;
  double odometry_translation_weight_;
  double odometry_rotation_weight_;
  ceres::Solver::Options ceres_options_;
};

class ImuBasedPoseExtrapolator : public PoseExtrapolatorInterface
{

public:
  explicit ImuBasedPoseExtrapolator();
  ~ImuBasedPoseExtrapolator() override;

  static std::unique_ptr<PoseExtrapolatorInterface>
  InitializeWithImu(const std::vector<sensor::ImuData>& imu_data,
                    const std::vector<TimestampedTransform>& initial_poses);

  // Returns the time of the last added pose or Time::min() if no pose was added
  // yet.
  double GetLastPoseTime() const override;
  double GetLastExtrapolatedTime() const override;

  void AddPose(double time, const Rigid3d& pose) override;
  void AddImuData(const sensor::ImuData& imu_data) override;
  void AddOdometryData(const sensor::OdometryData& odometry_data) override;

  Rigid3d ExtrapolatePose(double time) override;

  ExtrapolationResult
  ExtrapolatePosesWithGravity(const std::vector<double>& times) override;
  // Gravity alignment estimate.
  Eigen::Quaterniond EstimateGravityOrientation(double time) override;

private:
  template <typename T> void TrimDequeData(std::deque<T>* data);

  void TrimImuData();
  void TrimOdometryData();

  // Odometry methods.
  bool HasOdometryData() const;
  bool HasOdometryDataForTime(const double& first_time) const;
  TimestampedTransform Interpolate(const TimestampedTransform& start,
                                   const TimestampedTransform& end,
                                   const double& time);
  Rigid3d InterpolateOdometry(const double& first_time);
  Rigid3d
  CalculateOdometryBetweenNodes(const Rigid3d& first_node_odometry,
                                const Rigid3d& second_node_odometry) const;

  std::vector<Rigid3f>
  InterpolatePoses(const TimestampedTransform& start,
                   const TimestampedTransform& end,
                   const std::vector<double>::const_iterator times_begin,
                   const std::vector<double>::const_iterator times_end);

  std::vector<Rigid3f>
  InterpolatePoses(const TimestampedTransform& start,
                   const TimestampedTransform& end,
                   const std::multiset<double>::const_iterator times_begin,
                   const std::multiset<double>::const_iterator times_end);

  absl::flat_hash_map<float, Rigid3d> InterpolatePoses(
      const TimestampedTransform& start, const TimestampedTransform& end,
      const std::map<float, double, std::greater<float>>::const_iterator
          times_begin,
      const std::map<float, double, std::greater<float>>::const_iterator
          times_end);

  std::deque<TimestampedTransform> timed_pose_queue_;
  std::deque<TimestampedTransform> previous_solution_;

  std::deque<sensor::ImuData> imu_data_;
  std::deque<sensor::OdometryData> odometry_data_;
  double last_extrapolated_time_ = time_min();

  Rigid3d gravity_from_local_ = Rigid3d::Identity();

  //   const proto::ImuBasedPoseExtrapolatorOptions options_;
  const ceres::Solver::Options solver_options_;

  //   common::Histogram num_iterations_hist_;
};

#endif // CARTOGRAPHER_MAPPING_IMU_BASED_POSE_EXTRAPOLATOR_H_
