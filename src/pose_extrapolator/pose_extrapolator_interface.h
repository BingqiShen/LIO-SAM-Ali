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

#ifndef CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_INTERFACE_H_
#define CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_INTERFACE_H_

#include <memory>
#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "transform.h"
#include "sensor_data.h"
#include "imu_tracker.h"
#include <glog/logging.h>
class PoseExtrapolatorInterface
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  struct ExtrapolationResult
  {
    // The poses for the requested times at index 0 to N-1.
    std::vector<Rigid3f> previous_poses;
    // The pose for the requested time at index N.
    Rigid3d current_pose;
    Eigen::Vector3d current_velocity;
    Eigen::Quaterniond gravity_from_tracking;
  };

  struct ExtrapolationLegoResult
  {
    // The poses for the requested times at index 0 to N-1.
    absl::flat_hash_map<float, Rigid3d> poses_with_time;
    // The pose for the requested time at index N.
    Rigid3d current_pose;
    Eigen::Vector3d current_velocity;
    Eigen::Quaterniond gravity_from_tracking;
  };

  struct TimestampedTransform
  {
    double time;
    Rigid3d transform;
  };
  PoseExtrapolatorInterface(const PoseExtrapolatorInterface&) = delete;
  PoseExtrapolatorInterface&
  operator=(const PoseExtrapolatorInterface&) = delete;
  virtual ~PoseExtrapolatorInterface() {}

  // TODO: Remove dependency cycle.
  static std::unique_ptr<PoseExtrapolatorInterface>
  CreateWithImuData(const std::vector<sensor::ImuData>& imu_data,
                    const std::vector<TimestampedTransform>& initial_poses);

  // Returns the time of the last added pose or Time::min() if no pose was added
  // yet.
  virtual double GetLastPoseTime() const = 0;
  virtual double GetLastExtrapolatedTime() const = 0;

  virtual void AddPose(double time, const Rigid3d& pose) = 0;
  virtual void AddImuData(const sensor::ImuData& imu_data) = 0;
  virtual void AddOdometryData(const sensor::OdometryData& odometry_data) = 0;
  virtual Rigid3d ExtrapolatePose(double time) = 0;

  virtual ExtrapolationResult
  ExtrapolatePosesWithGravity(const std::vector<double>& times) = 0;
  // Returns the current gravity alignment estimate as a rotation from
  // the tracking frame into a gravity aligned frame.
  virtual Eigen::Quaterniond EstimateGravityOrientation(double time) = 0;
  static double time_min() { return 0.; }

protected:
  PoseExtrapolatorInterface() {}
};

#endif // CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_INTERFACE_H_
