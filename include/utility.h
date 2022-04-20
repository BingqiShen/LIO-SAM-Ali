#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2/convert.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <set>
#include <unordered_set>
#include "imu_tracker.h"
#include <glog/logging.h>
#include "transform.h"
#include "sensor_data.h"
#include <absl/memory/memory.h>
// #include <absl/strings/substitute.h>
using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType
{
  VELODYNE,
  OUSTER,
  HESAI
};

class ParamServer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ros::NodeHandle nh;

  std::string robot_id;

  // Topics
  string pointCloudTopic;
  string imuTopic;
  string odomTopic;
  string gpsTopic;

  // Frames
  string lidarFrame;
  string ImuFrame;
  string baselinkFrame;
  string odometryFrame;
  string mapFrame;

  // GPS Settings
  bool useImuHeadingInitialization;
  bool useGpsElevation;
  float gpsCovThreshold;
  float poseCovThreshold;

  // Save pcd
  bool savePCD;
  string savePCDDirectory;

  // Lidar Sensor Configuration
  SensorType sensor;
  int N_SCAN;
  int Horizon_SCAN;
  string timeField;
  int downsampleRate;
  float lidarMinRange;
  float lidarMaxRange;

  // IMU
  float imuAccNoise;
  float imuGyrNoise;
  float imuAccBiasN;
  float imuGyrBiasN;
  float imuGravity;
  float imuRPYWeight;
  vector<double> extRotV;
  vector<double> extRPYV;
  vector<double> extTransV;
  Eigen::Matrix3d imu2base_rot_;
  Eigen::Matrix3d extRPY;
  Eigen::Vector3d imu2base_t_;
  Eigen::Quaterniond extQRPY;
  Eigen::Quaterniond imu2base_q_;

  // LOAM
  float edgeThreshold;
  float surfThreshold;
  int edgeFeatureMinValidNum;
  int surfFeatureMinValidNum;

  // voxel filter paprams
  float odometrySurfLeafSize;
  float mappingCornerLeafSize;
  float mappingSurfLeafSize;

  float z_tollerance;
  float rotation_tollerance;

  // CPU Params
  int numberOfCores;
  double mappingProcessInterval;

  // Surrounding map
  float surroundingkeyframeAddingDistThreshold;
  float surroundingkeyframeAddingAngleThreshold;
  float surroundingKeyframeDensity;
  float surroundingKeyframeSearchRadius;
  float surroundingKeyframeSearchHeightRange;

  // Optimization
  float priorTranslationWeight;
  float priorRotationWeight;
  float secondOptimizationTranslationThreshold;
  float secondOptimizationRotationThreshold;

  // Loop closure
  bool loopClosureEnableFlag;
  float loopClosureFrequency;
  int surroundingKeyframeSize;
  float historyKeyframeSearchRadius;
  float historyKeyframeSearchTimeDiff;
  int historyKeyframeSearchNum;
  float historyKeyframeFitnessScore;

  // global map visualization radius
  float globalMapVisualizationSearchRadius;
  float globalMapVisualizationPoseDensity;
  float globalMapVisualizationLeafSize;
  tf2_ros::Buffer tfBuffer_;
  tf2_ros::TransformListener tfListener_;
  //     geometry_msgs::TransformStamped lidar2base_tf;
  geometry_msgs::TransformStamped imu2base_tf;
  //     geometry_msgs::TransformStamped imu2lidar_tf;
  Eigen::Quaterniond lidar2base_q_;
  int vtr_mode_;
  bool debug_;
  
  Eigen::Matrix3d eulToMatrix(Eigen::Vector3d eulur)
  {
    Eigen::Matrix3d matrix_tmp;
    matrix_tmp = (Eigen::AngleAxisd(eulur[0], Eigen::Vector3d::UnitZ()) *
                  Eigen::AngleAxisd(eulur[1], Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(eulur[2], Eigen::Vector3d::UnitX()))
                     .toRotationMatrix();
    return matrix_tmp;
  }
  ParamServer() : tfBuffer_(ros::Duration(20.)), tfListener_(tfBuffer_)
  {
    nh.param<std::string>("/robot_id", robot_id, "roboat");
    
    nh.param<std::string>("lio_sam/pointCloudTopic", pointCloudTopic,
                          "points_raw");
    nh.param<std::string>("lio_sam/imuTopic", imuTopic, "imu_correct");
    nh.param<std::string>("lio_sam/odomTopic", odomTopic, "odometry/imu");
    nh.param<std::string>("lio_sam/gpsTopic", gpsTopic, "odometry/gps");

    nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "lidar_link");
    nh.param<std::string>("lio_sam/ImuFrame", ImuFrame, "imu_link");
    nh.param<std::string>("lio_sam/baselinkFrame", baselinkFrame,
                          "base_footprint");
    nh.param<std::string>("lio_sam/odometryFrame", odometryFrame, "odom");
    nh.param<std::string>("lio_sam/mapFrame", mapFrame, "map");

    nh.param<bool>("lio_sam/useImuHeadingInitialization",
                   useImuHeadingInitialization, false);
    nh.param<bool>("lio_sam/useGpsElevation", useGpsElevation, false);
    nh.param<float>("lio_sam/gpsCovThreshold", gpsCovThreshold, 2.0);
    nh.param<float>("lio_sam/poseCovThreshold", poseCovThreshold, 25.0);

    nh.param<bool>("lio_sam/savePCD", savePCD, false);
    nh.param<std::string>("lio_sam/savePCDDirectory", savePCDDirectory,
                          "/Downloads/LOAM/");

    std::string sensorStr;
    nh.param<std::string>("lio_sam/sensor", sensorStr, "");
    nh.param<bool>("lio_sam/debug", debug_, false);
    cout << sensorStr << std::endl;
    if (sensorStr == "velodyne")
    {
      sensor = SensorType::VELODYNE;
    }
    else if (sensorStr == "ouster")
    {
      sensor = SensorType::OUSTER;
    }
    else if (sensorStr == "hesai")
    {
      sensor = SensorType::HESAI;
    }
    else
    {
      ROS_ERROR_STREAM(
          "Invalid sensor type (must be either 'velodyne' or 'ouster'): "
          << sensorStr);
      ros::shutdown();
    }

    nh.param<int>("lio_sam/N_SCAN", N_SCAN, 16);
    nh.param<int>("lio_sam/Horizon_SCAN", Horizon_SCAN, 1800);
    nh.param<int>("lio_sam/downsampleRate", downsampleRate, 1);
    nh.param<float>("lio_sam/lidarMinRange", lidarMinRange, 1.0);
    nh.param<float>("lio_sam/lidarMaxRange", lidarMaxRange, 1000.0);

    nh.param<float>("lio_sam/imuAccNoise", imuAccNoise, 0.01);
    nh.param<float>("lio_sam/imuGyrNoise", imuGyrNoise, 0.001);
    nh.param<float>("lio_sam/imuAccBiasN", imuAccBiasN, 0.0002);
    nh.param<float>("lio_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
    nh.param<float>("lio_sam/imuGravity", imuGravity, 9.80511);
    nh.param<float>("lio_sam/imuRPYWeight", imuRPYWeight, 0.01);
    nh.param<vector<double>>("lio_sam/extrinsicRot", extRotV, vector<double>());
    nh.param<vector<double>>("lio_sam/extrinsicRPY", extRPYV, vector<double>());
    nh.param<vector<double>>("lio_sam/extrinsicTrans", extTransV,
                             vector<double>());
    cout << "N_SCAN: " << N_SCAN << ", Horizon_SCAN: " << Horizon_SCAN << endl;

    Eigen::Vector3d extRPYV_eul(extRPYV[0], extRPYV[1], extRPYV[2]);
    extRPY = eulToMatrix(extRPYV_eul);
    extQRPY = Eigen::Quaterniond(extRPY);

    nh.param<float>("lio_sam/edgeThreshold", edgeThreshold, 0.1);
    nh.param<float>("lio_sam/surfThreshold", surfThreshold, 0.1);
    nh.param<int>("lio_sam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
    nh.param<int>("lio_sam/surfFeatureMinValidNum", surfFeatureMinValidNum,
                  100);
                
    nh.param<float>("lio_sam/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
    nh.param<float>("lio_sam/mappingCornerLeafSize", mappingCornerLeafSize,
                    0.2);
    nh.param<float>("lio_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

    nh.param<float>("lio_sam/z_tollerance", z_tollerance, FLT_MAX);
    nh.param<float>("lio_sam/rotation_tollerance", rotation_tollerance,
                    FLT_MAX);

    nh.param<int>("lio_sam/numberOfCores", numberOfCores, 2);
    nh.param<double>("lio_sam/mappingProcessInterval", mappingProcessInterval,
                     0);

    nh.param<float>("lio_sam/surroundingkeyframeAddingDistThreshold",
                    surroundingkeyframeAddingDistThreshold, 1.0);
    nh.param<float>("lio_sam/surroundingkeyframeAddingAngleThreshold",
                    surroundingkeyframeAddingAngleThreshold, 0.2);
    nh.param<float>("lio_sam/surroundingKeyframeDensity",
                    surroundingKeyframeDensity, 1.0);
    nh.param<float>("lio_sam/surroundingKeyframeSearchRadius",
                    surroundingKeyframeSearchRadius, 50.0);
    nh.param<float>("lio_sam/surroundingKeyframeSearchHeightRange",
                    surroundingKeyframeSearchHeightRange, 30.0);

    nh.param<float>("lio_sam/priorTranslationWeight",
                    priorTranslationWeight, 0.0);
    nh.param<float>("lio_sam/priorRotationWeight",
                    priorRotationWeight, 0.0);
    nh.param<float>("lio_sam/secondOptimizationTranslationThreshold",
                    secondOptimizationTranslationThreshold, 10.0);
    nh.param<float>("lio_sam/secondOptimizationRotationThreshold",
                    secondOptimizationRotationThreshold, 1.0);

    nh.param<bool>("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag,
                   false);
    nh.param<float>("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0);
    nh.param<int>("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize,
                  50);
    nh.param<float>("lio_sam/historyKeyframeSearchRadius",
                    historyKeyframeSearchRadius, 10.0);
    nh.param<float>("lio_sam/historyKeyframeSearchTimeDiff",
                    historyKeyframeSearchTimeDiff, 30.0);
    nh.param<int>("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum,
                  25);
    nh.param<float>("lio_sam/historyKeyframeFitnessScore",
                    historyKeyframeFitnessScore, 0.3);

    nh.param<float>("lio_sam/globalMapVisualizationSearchRadius",
                    globalMapVisualizationSearchRadius, 1e3);
    nh.param<float>("lio_sam/globalMapVisualizationPoseDensity",
                    globalMapVisualizationPoseDensity, 10.0);
    nh.param<float>("lio_sam/globalMapVisualizationLeafSize",
                    globalMapVisualizationLeafSize, 1.0);
    vtr_mode_ = 1;
    ros::param::get("/vtr/mode", vtr_mode_);
    int kk = 0;
    bool use_sim_time = false;
    ros::param::get("/use_sim_time", use_sim_time);
    while (1)
    {
      try
      {
        //             lidar2base_tf = tfBuffer_.lookupTransform(baselinkFrame,
        //             lidarFrame, ros::Time(0));
        imu2base_tf =
            tfBuffer_.lookupTransform(baselinkFrame, ImuFrame, ros::Time(0));
//         LOG(WARNING) << 
        //             imu2lidar_tf = tfBuffer_.lookupTransform(lidarFrame,
        //             ImuFrame, ros::Time(0));
        break;
      }
      catch (tf::TransformException ex)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if ( kk++ > 50)
        {
          LOG(WARNING) << baselinkFrame <<", " << ImuFrame;
          LOG(WARNING) << "exit, no tf_static pub!";
          exit(0);
        }
      }
    }

    imu2base_q_ = Eigen::Quaterniond(
        imu2base_tf.transform.rotation.w, imu2base_tf.transform.rotation.x,
        imu2base_tf.transform.rotation.y, imu2base_tf.transform.rotation.z);
    imu2base_rot_ = imu2base_q_.toRotationMatrix();
    imu2base_t_ = Eigen::Vector3d(imu2base_tf.transform.translation.x,
                                  imu2base_tf.transform.translation.y,
                                  imu2base_tf.transform.translation.z);
    cout << "imu2lidar_t_: " << imu2base_t_.transpose() << endl;
    cout << "extRot: " << imu2base_rot_ << endl;
    usleep(100);
  }

  sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
  {
    sensor_msgs::Imu imu_out = imu_in;
    // rotate acceleration
    Eigen::Vector3d acc(imu_in.linear_acceleration.x,
                        imu_in.linear_acceleration.y,
                        imu_in.linear_acceleration.z);
    acc = imu2base_rot_ * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();
    // rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y,
                        imu_in.angular_velocity.z);
    gyr = imu2base_rot_ * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();
    // rotate roll pitch yaw
    Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x,
                              imu_in.orientation.y, imu_in.orientation.z);
    Eigen::Quaterniond q_final = q_from * extQRPY;
    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() +
             q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1)
    {
      ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
      ros::shutdown();
    }

    return imu_out;
  }

  sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in,
                                std::shared_ptr<ImuTracker> imu_tracker)
  {
    sensor_msgs::Imu imu_out = imu_in;
//     if(sqrt(imu_out.orientation.x * imu_out.orientation.x + 
//       imu_out.orientation.y * imu_out.orientation.y +
//       imu_out.orientation.z * imu_out.orientation.z +
//       imu_out.orientation.w * imu_out.orientation.w) < 0.1)
    {
      Eigen::Vector3d imu_linear_acceleration(imu_in.linear_acceleration.x, 
                          imu_in.linear_acceleration.y, 
                          imu_in.linear_acceleration.z);
      Eigen::Vector3d imu_angular_velocity(imu_in.angular_velocity.x, 
                          imu_in.angular_velocity.y, 
                          imu_in.angular_velocity.z);
      imu_linear_acceleration = imu2base_q_ * imu_linear_acceleration;
      imu_angular_velocity = imu2base_q_ *imu_angular_velocity;
      imu_out.linear_acceleration.x = imu_linear_acceleration.x();
      imu_out.linear_acceleration.y = imu_linear_acceleration.y();
      imu_out.linear_acceleration.z = imu_linear_acceleration.z();
      
      imu_out.angular_velocity.x = imu_angular_velocity.x();
      imu_out.angular_velocity.y = imu_angular_velocity.y();
      imu_out.angular_velocity.z = imu_angular_velocity.z();
        
      imu_tracker->Advance(imu_in.header.stamp.toSec());
      imu_tracker->AddImuLinearAccelerationObservation(imu_linear_acceleration);
      imu_tracker->AddImuAngularVelocityObservation(imu_angular_velocity);
//       Eigen::Quaterniond q_final = imu2base_q_ * imu_tracker->orientation();
      Eigen::Quaterniond q_final = imu_tracker->orientation();
      imu_out.orientation.x = q_final.x();
      imu_out.orientation.y = q_final.y();
      imu_out.orientation.z = q_final.z();
      imu_out.orientation.w = q_final.w();
      
//       LOG(WARNING) << "imu tracker orientation: " << q_final.toRotationMatrix();
//       
//       Eigen::Quaterniond q_tmp(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
//       LOG(WARNING) << "origin orientation: " << q_tmp.toRotationMatrix();
      if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
      {
          ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
          ros::shutdown();
      }
    }
    return imu_out;
  }

  inline Eigen::Vector3d ToEigen(const geometry_msgs::Vector3& vector3)
  {
    return Eigen::Vector3d(vector3.x, vector3.y, vector3.z);
  }

  inline Eigen::Quaterniond ToEigen(const geometry_msgs::Quaternion& quaternion)
  {
    return Eigen::Quaterniond(quaternion.w, quaternion.x, quaternion.y,
                              quaternion.z);
  }
  inline Rigid3d ToRigid3d(const geometry_msgs::TransformStamped& transform)
  {
    return Rigid3d(ToEigen(transform.transform.translation),
                   ToEigen(transform.transform.rotation));
  }

  inline Rigid3d ToRigid3d(const geometry_msgs::Pose& pose)
  {
    return Rigid3d({pose.position.x, pose.position.y, pose.position.z},
                   ToEigen(pose.orientation));
  }
  sensor::ImuData ToImuData(const sensor_msgs::Imu::ConstPtr& msg)
  {
    const double time = msg->header.stamp.toSec();
    Eigen::Vector3d linear_acceleration = 
      imu2base_q_ * ToEigen(msg->linear_acceleration);
    Eigen::Vector3d angular_velocity =
      imu2base_q_ * ToEigen(msg->angular_velocity);
    return sensor::ImuData{time, linear_acceleration, angular_velocity};
  }

  sensor::OdometryData ToOdometryData(const nav_msgs::Odometry::ConstPtr& msg)
  {
    const double time = msg->header.stamp.toSec();
    return sensor::OdometryData{
        time, ToRigid3d(msg->pose.pose) /** sensor_to_tracking->inverse()*/,
        Eigen::Vector3d(msg->twist.twist.linear.x, msg->twist.twist.linear.y,
                        msg->twist.twist.linear.z),
        Eigen::Vector3d(msg->twist.twist.angular.x, msg->twist.twist.angular.y,
                        msg->twist.twist.angular.z)};
  }
};
class VoxelFilter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using KeyType = std::bitset<3 * 32>;
  VoxelFilter(const float& resolution) : resolution_(resolution) {}

  KeyType IndexToKey(const Eigen::Array3i& index)
  {
    KeyType k_0(static_cast<uint32_t>(index[0]));
    KeyType k_1(static_cast<uint32_t>(index[1]));
    KeyType k_2(static_cast<uint32_t>(index[2]));
    return (k_0 << 2 * 32) | (k_1 << 1 * 32) | k_2;
  }

  Eigen::Array3i GetCellIndex(const Eigen::Vector3f& point) const
  {
    Eigen::Array3f index = point.array() / resolution_;
    return Eigen::Array3i(std::round(index.x()), std::round(index.y()),
                          std::round(index.z()));
  }

  //   template <typename T>
  pcl::PointCloud<PointType>::Ptr
  Filter(const pcl::PointCloud<PointType>::Ptr& cloud)
  {
    pcl::PointCloud<PointType>::Ptr results(new pcl::PointCloud<PointType>());

    std::unordered_set<KeyType> voxel_set_;
    for (const PointType& point : cloud->points)
    {
      auto it_inserted = voxel_set_.insert(
          IndexToKey(GetCellIndex({point.x, point.y, point.z})));
      if (it_inserted.second)
      {
        results->push_back(point);
      }
    }
    return results;
  }

private:
  const float resolution_;
};

template <typename T> double ROS_TIME(T msg)
{
  return msg->header.stamp.toSec();
}

template <typename T>
void imuAngular2rosAngular(sensor_msgs::Imu* thisImuMsg, T* angular_x,
                           T* angular_y, T* angular_z)
{
  *angular_x = thisImuMsg->angular_velocity.x;
  *angular_y = thisImuMsg->angular_velocity.y;
  *angular_z = thisImuMsg->angular_velocity.z;
}

template <typename T>
void imuAccel2rosAccel(sensor_msgs::Imu* thisImuMsg, T* acc_x, T* acc_y,
                       T* acc_z)
{
  *acc_x = thisImuMsg->linear_acceleration.x;
  *acc_y = thisImuMsg->linear_acceleration.y;
  *acc_z = thisImuMsg->linear_acceleration.z;
}

template <typename T>
void imuRPY2rosRPY(sensor_msgs::Imu* thisImuMsg, T* rosRoll, T* rosPitch,
                   T* rosYaw)
{
  double imuRoll, imuPitch, imuYaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

  *rosRoll = imuRoll;
  *rosPitch = imuPitch;
  *rosYaw = imuYaw;
}

#endif
