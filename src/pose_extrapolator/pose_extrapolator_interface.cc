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

#include "pose_extrapolator_interface.h"

// #include "cartographer/common/ceres_solver_options.h"
// #include "cartographer/common/time.h"
#include "imu_based_pose_extrapolator.h"
#include "pose_extrapolator.h"

std::unique_ptr<PoseExtrapolatorInterface>
PoseExtrapolatorInterface::CreateWithImuData(
    const std::vector<sensor::ImuData>& imu_data,
    const std::vector<TimestampedTransform>& initial_poses)
{
  CHECK(!imu_data.empty());

  bool use_imu_based_pose_extrapolator_ = false;
  if (use_imu_based_pose_extrapolator_)
  {
    return ImuBasedPoseExtrapolator::InitializeWithImu(imu_data, initial_poses);
  }
  else
  {
    return PoseExtrapolator::InitializeWithImu(0.03, 1, imu_data.back());
  }
}
