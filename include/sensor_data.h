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

#ifndef SENSOR_DATA_H_
#define SENSOR_DATA_H_

#include "Eigen/Core"
#include "transform.h"
namespace sensor
{
struct ImuData
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double time;
  Eigen::Vector3d linear_acceleration;
  Eigen::Vector3d angular_velocity;
};

struct OdometryData
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double time;
  Rigid3d pose;
  Eigen::Vector3d twist_linear;
  Eigen::Vector3d twist_angular;
};

} // namespace sensor

#endif // SENSOR_DATA_H_
