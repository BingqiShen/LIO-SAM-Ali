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

#include "imu_based_pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
// #include "cartographer/mapping/eigen_quaterniond_from_two_vectors.h"
// #include "cartographer/mapping/internal/3d/imu_integration.h"
// #include "cartographer/mapping/internal/3d/rotation_parameterization.h"
#include "ceres_pose.h"
#include "cost_functions/acceleration_cost_function_3d.h"
#include "cost_functions/rotation_cost_function_3d.h"
#include "cost_functions/spa_cost_function_3d.h"
#include "imu_integration.h"
// #include "cartographer/mapping/pose_graph_interface.h"
// #include "cartographer/transform/transform.h"
#include "glog/logging.h"
#include "ros/ros.h"

// using ::cartographer::transform::TimestampedTransform;
ImuBaseOptions::ImuBaseOptions()
{

  pose_queue_duration_ = 0.5;
  gravity_constant_ = 9.806;
  pose_translation_weight_ = 1e5;
  pose_rotation_weight_ = 1e5;
  imu_acceleration_weight_ = 1e-9;
  imu_rotation_weight_ = 1e2;
  odometry_translation_weight_ = 1e5;
  odometry_rotation_weight_ = 1e-1;
  bool use_nonmonotonic_steps = false;
  int max_num_iterations = 10;
  int num_threads = 1;

  if (!ros::param::get("/imu_based_options/pose_queue_duration",
                       pose_queue_duration_))
    LOG(WARNING) << "Can not get imu_based_options params!";

  ros::param::get("/imu_based_options/gravity_constant", gravity_constant_);
  ros::param::get("/imu_based_options/pose_translation_weight",
                  pose_translation_weight_);
  ros::param::get("/imu_based_options/pose_rotation_weight",
                  pose_rotation_weight_);
  ros::param::get("/imu_based_options/imu_acceleration_weight",
                  imu_acceleration_weight_);
  ros::param::get("/imu_based_options/imu_rotation_weight",
                  imu_rotation_weight_);
  ros::param::get("/imu_based_options/odometry_translation_weight",
                  odometry_translation_weight_);
  ros::param::get("/imu_based_options/odometry_rotation_weight",
                  odometry_rotation_weight_);
  ros::param::get(
      "/imu_based_options/ceres_solver_options/use_nonmonotonic_steps",
      use_nonmonotonic_steps);
  ros::param::get("/imu_based_options/ceres_solver_options/max_num_iterations",
                  max_num_iterations);
  ros::param::get("/imu_based_options/ceres_solver_options/num_threads",
                  num_threads);

  LOG(INFO) << "pose_queue_duration: " << pose_queue_duration_;
  LOG(INFO) << "gravity_constant: " << gravity_constant_;
  LOG(INFO) << "pose_translation_weight: " << pose_translation_weight_;
  LOG(INFO) << "pose_rotation_weight: " << pose_rotation_weight_;
  LOG(INFO) << "imu_acceleration_weight: " << imu_acceleration_weight_;
  LOG(INFO) << "imu_rotation_weight: " << imu_rotation_weight_;
  LOG(INFO) << "odometry_translation_weight: " << odometry_translation_weight_;
  LOG(INFO) << "odometry_rotation_weight: " << odometry_rotation_weight_;
  LOG(INFO) << "ceres_solver_options/use_nonmonotonic_steps: "
            << use_nonmonotonic_steps;
  LOG(INFO) << "ceres_solver_options/max_num_iterations: "
            << max_num_iterations;
  LOG(INFO) << "ceres_solver_options/num_threads: " << num_threads;

  ceres_options_.use_nonmonotonic_steps = use_nonmonotonic_steps;
  ceres_options_.max_num_iterations = max_num_iterations;
  ceres_options_.num_threads = num_threads;
}

ImuBasedPoseExtrapolator::ImuBasedPoseExtrapolator()
    : solver_options_(ImuBaseOptions::options().ceres_options())
{
}

ImuBasedPoseExtrapolator::~ImuBasedPoseExtrapolator()
{
  LOG(INFO) << "Number of iterations for pose extrapolation:";
  //   LOG(INFO) << num_iterations_hist_.ToString(10);
}

std::unique_ptr<PoseExtrapolatorInterface>
ImuBasedPoseExtrapolator::InitializeWithImu(
    const std::vector<sensor::ImuData>& imu_data,
    const std::vector<TimestampedTransform>& initial_poses)
{
  CHECK(!imu_data.empty());
  auto extrapolator = absl::make_unique<ImuBasedPoseExtrapolator>();
  std::copy(imu_data.begin(), imu_data.end(),
            std::back_inserter(extrapolator->imu_data_));
  if (!initial_poses.empty())
  {
    for (const auto& pose : initial_poses)
    {
      if (pose.time > imu_data.front().time)
      {
        extrapolator->AddPose(pose.time, pose.transform);
      }
    }
  }
  else
  {
    extrapolator->AddPose(
        imu_data.back().time,
        Rigid3d::Rotation(Eigen::Quaterniond::FromTwoVectors(
            imu_data.back().linear_acceleration, Eigen::Vector3d::UnitZ())));
  }
  return extrapolator;
}

double ImuBasedPoseExtrapolator::GetLastPoseTime() const
{
  if (timed_pose_queue_.empty())
  {
    return time_min();
  }
  return timed_pose_queue_.back().time;
}

double ImuBasedPoseExtrapolator::GetLastExtrapolatedTime() const
{
  return last_extrapolated_time_;
}

void ImuBasedPoseExtrapolator::AddPose(const double time, const Rigid3d& pose)
{
  timed_pose_queue_.push_back(TimestampedTransform{time, pose});
  while (timed_pose_queue_.size() > 3 &&
         timed_pose_queue_[1].time <=
             time - ImuBaseOptions::options().pose_queue_duration())
  {
    if (!previous_solution_.empty())
    {
      CHECK_EQ(timed_pose_queue_.front().time, previous_solution_.front().time);
      previous_solution_.pop_front();
    }
    timed_pose_queue_.pop_front();
  }
  TrimImuData();
}

void ImuBasedPoseExtrapolator::AddImuData(const sensor::ImuData& imu_data)
{
  CHECK(timed_pose_queue_.empty() ||
        imu_data.time >= timed_pose_queue_.back().time);
  imu_data_.push_back(imu_data);
  TrimImuData();
}

void ImuBasedPoseExtrapolator::AddOdometryData(
    const sensor::OdometryData& odometry_data)
{
  CHECK(timed_pose_queue_.empty() ||
        odometry_data.time >= timed_pose_queue_.back().time);
  odometry_data_.push_back(odometry_data);
  TrimOdometryData();
}

ImuBasedPoseExtrapolator::ExtrapolationResult
ImuBasedPoseExtrapolator::ExtrapolatePosesWithGravity(
    const std::vector<double>& times)
{
  const auto& time = times.back();
  const auto& newest_timed_pose = timed_pose_queue_.back();
  CHECK_GE(time, newest_timed_pose.time);
  CHECK_GE(times.size(), 1);
  last_extrapolated_time_ = time;
  //   LOG(WARNING) << "newest_timed_pose time: " <<std::setprecision(16) <<
  //   newest_timed_pose.time << ", " << timed_pose_queue_.size() <<", "
  //   <<previous_solution_.size();
  //   if(!previous_solution_.empty())
  //   LOG(WARNING) << "previous_solution_ time: " <<std::setprecision(16) <<
  //   previous_solution_.back().time;
  if (timed_pose_queue_.size() < 3 || (time - newest_timed_pose.time) < 1e-6)
  {
    return ExtrapolationResult{
        std::vector<Rigid3f>(times.size() - 1,
                             timed_pose_queue_.back().transform.cast<float>()),
        timed_pose_queue_.back().transform, Eigen::Vector3d::Zero(),
        timed_pose_queue_.back().transform.rotation()};
  }

  ceres::Problem::Options problem_options;
  ceres::Problem problem(problem_options);

  // Track gravity alignment over time and use this as a frame here so that
  // we can estimate the gravity alignment of the current pose.
  CeresPose gravity_from_local(
      gravity_from_local_, nullptr,
      absl::make_unique<ceres::QuaternionParameterization>(), &problem);
  // Use deque so addresses stay constant during problem formulation.
  std::deque<CeresPose> nodes;
  std::vector<double> node_times;
  for (size_t i = 0; i < timed_pose_queue_.size(); i++)
  {
    bool is_last = (i == (timed_pose_queue_.size() - 1));
    const auto& timed_pose = timed_pose_queue_[i];

    node_times.push_back(timed_pose.time);
    Rigid3d gravity_from_node;
    // Use the last scan match result (timed_pose_queue_.back()) for
    // initialization here instead of the last result from the optimization.
    // This keeps poses from slowly drifting apart due to lack of feedback
    // from the scan matching here.
    if ((!is_last && previous_solution_[i].time > timed_pose.time) ||
        previous_solution_.size() <=
            i) // protect it if the node addpose twice continuously
      is_last = true;
    if (!previous_solution_.empty() && !is_last)
    {
      CHECK_GT(previous_solution_.size(), i);
      CHECK_EQ(timed_pose.time, previous_solution_[i].time);
      gravity_from_node = previous_solution_[i].transform;
    }
    else
    {
      gravity_from_node = gravity_from_local_ * timed_pose.transform;
    }

    if (is_last)
    {
      nodes.emplace_back(gravity_from_node, nullptr,
                         absl::make_unique<ceres::AutoDiffLocalParameterization<
                             ConstantYawQuaternionPlus, 4, 2>>(),
                         &problem);
      problem.SetParameterBlockConstant(nodes.back().translation());
    }
    else
    {
      nodes.emplace_back(gravity_from_node, nullptr,
                         absl::make_unique<ceres::QuaternionParameterization>(),
                         &problem);
    }
  }
  double gravity_constant = 9.8;
  bool fix_gravity = false;
  if (ImuBaseOptions::options().gravity_constant() > 0)
  {
    fix_gravity = true;
    gravity_constant = ImuBaseOptions::options().gravity_constant();
  }
  auto imu_it_prev_prev = imu_data_.begin();
  while (imu_it_prev_prev != std::prev(imu_data_.end()) &&
         std::next(imu_it_prev_prev)->time <= timed_pose_queue_.back().time)
  {
    ++imu_it_prev_prev;
  }

  const TimestampedTransform prev_gravity_from_tracking =
      TimestampedTransform{node_times.back(), nodes.back().ToRigid()};
  const TimestampedTransform prev_prev_gravity_from_tracking =
      TimestampedTransform{node_times.at(node_times.size() - 2),
                           nodes.at(nodes.size() - 2).ToRigid()};
  const Rigid3d initial_estimate =
      ExtrapolatePoseWithImu<double>(
          prev_gravity_from_tracking.transform, prev_gravity_from_tracking.time,
          prev_prev_gravity_from_tracking.transform,
          prev_prev_gravity_from_tracking.time,
          gravity_constant * Eigen::Vector3d::UnitZ(), time, imu_data_,
          &imu_it_prev_prev).pose;
  nodes.emplace_back(initial_estimate, nullptr,
                     absl::make_unique<ceres::QuaternionParameterization>(),
                     &problem);
  node_times.push_back(time);
  // Add cost functions for node constraints.
  for (size_t i = 0; i < timed_pose_queue_.size(); i++)
  {
    const auto& timed_pose = timed_pose_queue_[i];
    problem.AddResidualBlock(
        SpaCostFunction3D::CreateAutoDiffCostFunction(
            ConstraintPose{timed_pose.transform,
                           ImuBaseOptions::options().pose_translation_weight(),
                           ImuBaseOptions::options().pose_rotation_weight()}),
        nullptr /* loss function */, gravity_from_local.rotation(),
        gravity_from_local.translation(), nodes.at(i).rotation(),
        nodes.at(i).translation());
  }
  CHECK(!imu_data_.empty());
  CHECK_LE(imu_data_.front().time, timed_pose_queue_.front().time);

  std::array<double, 4> imu_calibration{{1., 0., 0., 0.}};

  problem.AddParameterBlock(imu_calibration.data(), 4,
                            new ceres::QuaternionParameterization());
  problem.SetParameterBlockConstant(imu_calibration.data());
  auto imu_it = imu_data_.begin();
  CHECK(imu_data_.size() == 1 ||
        std::next(imu_it)->time > timed_pose_queue_.front().time);

  Rigid3d last_node_odometry;
  double last_node_odometry_time;
  for (size_t i = 1; i < nodes.size(); i++)
  {
    const double first_time = node_times[i - 1];
    const double second_time = node_times[i];

    auto imu_it2 = imu_it;
    const IntegrateImuResult<double> result =
        IntegrateImu(imu_data_, first_time, second_time, &imu_it);
    if ((i + 1) < nodes.size() && 0)
    {
      const double third_time = node_times[i + 1];
      const double first_duration = second_time - first_time;
      const double second_duration = third_time - second_time;
      const double first_center = first_time + first_duration / 2;
      const double second_center = second_time + second_duration / 2;
      const IntegrateImuResult<double> result_to_first_center =
          IntegrateImu(imu_data_, first_time, first_center, &imu_it2);
      const IntegrateImuResult<double> result_center_to_center =
          IntegrateImu(imu_data_, first_center, second_center, &imu_it2);
      // 'delta_velocity' is the change in velocity from the point in time
      // halfway between the first and second poses to halfway between
      // second and third pose. It is computed from IMU data and still
      // contains a delta due to gravity. The orientation of this vector is
      // in the IMU frame at the second pose.
      const Eigen::Vector3d delta_velocity =
          (result.delta_rotation.inverse() *
           result_to_first_center.delta_rotation) *
          result_center_to_center.delta_velocity;
      problem.AddResidualBlock(
          AccelerationCostFunction3D::CreateAutoDiffCostFunction(
              ImuBaseOptions::options().imu_acceleration_weight(),
              delta_velocity, (first_duration), (second_duration)),
          nullptr /* loss function */, nodes.at(i).rotation(),
          nodes.at(i - 1).translation(), nodes.at(i).translation(),
          nodes.at(i + 1).translation(), &gravity_constant,
          imu_calibration.data());
      // TODO(danielsievers): Fix gravity in CostFunction.
      if (fix_gravity)
      {
        problem.SetParameterBlockConstant(&gravity_constant);
      }
      else
      {
        // Force gravity constant to be positive.
        problem.SetParameterLowerBound(&gravity_constant, 0, 0.0);
      }
    }

    problem.AddResidualBlock(
        RotationCostFunction3D::CreateAutoDiffCostFunction(
            ImuBaseOptions::options().imu_rotation_weight(),
            result.delta_rotation),
        nullptr /* loss function */, nodes.at(i - 1).rotation(),
        nodes.at(i).rotation(), imu_calibration.data());
    //     problem.SetParameterBlockConstant(imu_calibration.data());
    // Add a relative pose constraint based on the odometry (if available).
    if (HasOdometryDataForTime(first_time) &&
        HasOdometryDataForTime(second_time))
    {
      // Here keep track of last node odometry to avoid double computation.
      // Do this if first loop, or if there were some missing odometry nodes
      // then recalculate.
      if (i == 1 || last_node_odometry_time != first_time)
      {
        last_node_odometry = InterpolateOdometry(first_time);
        last_node_odometry_time = first_time;
      }
      const Rigid3d current_node_odometry = InterpolateOdometry(second_time);
      const Rigid3d relative_odometry = CalculateOdometryBetweenNodes(
          last_node_odometry, current_node_odometry);

      problem.AddResidualBlock(
          SpaCostFunction3D::CreateAutoDiffCostFunction(ConstraintPose{
              relative_odometry,
              ImuBaseOptions::options().odometry_translation_weight(),
              ImuBaseOptions::options().odometry_rotation_weight()}),
          nullptr /* loss function */, nodes.at(i - 1).rotation(),
          nodes.at(i - 1).translation(), nodes.at(i).rotation(),
          nodes.at(i).translation());
      // Use the current node odometry in the next iteration
      last_node_odometry = current_node_odometry;
      last_node_odometry_time = second_time;
    }
  }
  // Solve.
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options_, &problem, &summary);
  LOG_IF_EVERY_N(INFO, !fix_gravity, 20) << "Gravity was: " << gravity_constant;

  const auto gravity_estimate = nodes.back().ToRigid().rotation();

  const auto& last_pose = timed_pose_queue_.back();
  const auto extrapolated_pose = TimestampedTransform{
      time, last_pose.transform *
                nodes.at(nodes.size() - 2).ToRigid().inverse() *
                nodes.back().ToRigid()};
  //   num_iterations_hist_.Add(summary.iterations.size());

  gravity_from_local_ = gravity_from_local.ToRigid();
  LOG_EVERY_N(WARNING, 50) << gravity_from_local_;
  previous_solution_.clear();
  for (size_t i = 0; i < nodes.size(); ++i)
  {
    previous_solution_.push_back(
        TimestampedTransform{node_times.at(i), nodes.at(i).ToRigid()});
  }
  const Eigen::Vector3d current_velocity =
      (extrapolated_pose.transform.translation() -
       last_pose.transform.translation()) /
      (time - last_pose.time);

  return ExtrapolationResult{
      InterpolatePoses(last_pose, extrapolated_pose, times.begin(),
                       std::prev(times.end())),
      extrapolated_pose.transform, current_velocity, gravity_estimate};
}

std::vector<Rigid3f> ImuBasedPoseExtrapolator::InterpolatePoses(
    const TimestampedTransform& start, const TimestampedTransform& end,
    const std::vector<double>::const_iterator times_begin,
    const std::vector<double>::const_iterator times_end)
{
  std::vector<Rigid3f> poses;
  poses.reserve(std::distance(times_begin, times_end));
  const float duration_scale = 1. / (end.time - start.time);

  const Eigen::Quaternionf start_rotation =
      Eigen::Quaternionf(start.transform.rotation());
  const Eigen::Quaternionf end_rotation =
      Eigen::Quaternionf(end.transform.rotation());
  const Eigen::Vector3f start_translation =
      start.transform.translation().cast<float>();
  const Eigen::Vector3f end_translation =
      end.transform.translation().cast<float>();

  for (auto it = times_begin; it != times_end; ++it)
  {
    const float factor = (*it - start.time) * duration_scale;
    const Eigen::Vector3f origin =
        start_translation + (end_translation - start_translation) * factor;
    const Eigen::Quaternionf rotation =
        start_rotation.slerp(factor, end_rotation);
    poses.emplace_back(origin, rotation);
  }
  return poses;
}

Rigid3d ImuBasedPoseExtrapolator::ExtrapolatePose(const double time)
{
  return ExtrapolatePosesWithGravity(std::vector<double>{time}).current_pose;
}

Eigen::Quaterniond
ImuBasedPoseExtrapolator::EstimateGravityOrientation(const double time)
{
  return ExtrapolatePosesWithGravity(std::vector<double>{time})
      .gravity_from_tracking;
}

template <typename T>
void ImuBasedPoseExtrapolator::TrimDequeData(std::deque<T>* data)
{
  while (data->size() > 1 && !timed_pose_queue_.empty() &&
         data->at(1).time <= timed_pose_queue_.front().time)
  {
    data->pop_front();
  }
}

void ImuBasedPoseExtrapolator::TrimImuData()
{
  TrimDequeData<sensor::ImuData>(&imu_data_);
}

void ImuBasedPoseExtrapolator::TrimOdometryData()
{
  TrimDequeData<sensor::OdometryData>(&odometry_data_);
}

// Odometry methods
bool ImuBasedPoseExtrapolator::HasOdometryData() const
{
  return odometry_data_.size() >= 2;
}

bool ImuBasedPoseExtrapolator::HasOdometryDataForTime(const double& time) const
{
  return HasOdometryData() && odometry_data_.front().time < time &&
         time < odometry_data_.back().time;
}

ImuBasedPoseExtrapolator::TimestampedTransform
ImuBasedPoseExtrapolator::Interpolate(const TimestampedTransform& start,
                                      const TimestampedTransform& end,
                                      const double& time)
{
  CHECK_LE(start.time, time);
  CHECK_GE(end.time, time);

  const double duration = (end.time - start.time);
  const double factor = (time - start.time) / duration;
  const Eigen::Vector3d origin =
      start.transform.translation() +
      (end.transform.translation() - start.transform.translation()) * factor;
  const Eigen::Quaterniond rotation =
      Eigen::Quaterniond(start.transform.rotation())
          .slerp(factor, Eigen::Quaterniond(end.transform.rotation()));
  return TimestampedTransform{time, Rigid3d(origin, rotation)};
}

Rigid3d ImuBasedPoseExtrapolator::InterpolateOdometry(const double& time)
{
  // Only interpolate if time is within odometry data range.
  CHECK(HasOdometryDataForTime(time))
      << "Odometry data range does not include time " << time;
  std::deque<sensor::OdometryData>::const_iterator data = std::upper_bound(
      odometry_data_.begin(), odometry_data_.end(), time,
      [](const double& time, const sensor::OdometryData& odometry_data)
      {
        return time < odometry_data.time;
      });
  const TimestampedTransform first{std::prev(data)->time,
                                   std::prev(data)->pose};
  const TimestampedTransform second{data->time, data->pose};
  return Interpolate(first, second, time).transform;
}

Rigid3d ImuBasedPoseExtrapolator::CalculateOdometryBetweenNodes(
    const Rigid3d& first_node_odometry,
    const Rigid3d& second_node_odometry) const
{
  return first_node_odometry.inverse() * second_node_odometry;
}
