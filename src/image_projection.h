#include "utility.h"
#include "lio_sam/cloud_info.h"
#include <unordered_map>

struct VelodynePointXYZIRT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    VelodynePointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                            intensity)(uint16_t, ring,
                                                       ring)(float, time, time))

struct OusterPointXYZIRT
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t ring;
  uint16_t noise;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    OusterPointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        uint32_t, t, t)(uint16_t, reflectivity, reflectivity)(
        uint8_t, ring, ring)(uint16_t, noise, noise)(uint32_t, range, range))

// hesai(pandar)
struct HesaiPointXYZIRT
{
  PCL_ADD_POINT4D;
  uint8_t intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(
    HesaiPointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)(
        double, timestamp, timestamp)(uint16_t, ring, ring))
struct smoothness_t
{
  float value;
  size_t ind;
};

struct by_value
{
  bool operator()(smoothness_t const& left, smoothness_t const& right)
  {
    return left.value < right.value;
  }
};
// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;
// using PointXYZIRT = HesaiPointXYZIRT;

const int queueLength = 2000;
class ImageProjection : public ParamServer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ImageProjection();
  ~ImageProjection();
  void allocateMemory();
  void resetParameters();
  void cloudHandler(const pcl::PointCloud<PointXYZIRT>::Ptr& cloud,
                    std::unordered_map<double, Eigen::Affine3f> points_pose,
                    const bool& imu_valid, const bool& odom_valid, 
                    const Eigen::Affine3f& lidar2base);
  PointType deskewPoint(PointType* point, const Eigen::Affine3f& relTime);
  void
  projectPointCloud(const pcl::PointCloud<PointXYZIRT>::Ptr& laserCloudIn,
                    std::unordered_map<double, Eigen::Affine3f> points_pose,
                    const Eigen::Affine3f& lidar2base);
  void cloudExtraction();

  void laserCloudInfoHandler();
  void calculateSmoothness();
  void markOccludedPoints();
  void extractFeatures();
  void freeCloudInfoMemory();
  void getFeatureCloud(const Eigen::Affine3f& lidar2base,
                       pcl::PointCloud<PointType>::Ptr& corner_cloud,
                       pcl::PointCloud<PointType>::Ptr& surface_cloud,
                       std::shared_ptr<lio_sam::cloud_info>& cloud_info);

private:
  std::mutex imuLock;
  std::mutex odoLock;

  std::deque<sensor_msgs::PointCloud2> cloudQueue;
  sensor_msgs::PointCloud2 currentCloudMsg;
  int imuPointerCur;
  bool firstPointFlag;
  Eigen::Affine3f transStartInverse;

  pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
  pcl::PointCloud<HesaiPointXYZIRT>::Ptr tmpHesaiCloudIn;
  pcl::PointCloud<PointType>::Ptr fullCloud;
  pcl::PointCloud<PointType>::Ptr extractedCloud;
  pcl::PointCloud<PointType>::Ptr cornerCloud;
  pcl::PointCloud<PointType>::Ptr surfaceCloud;
  pcl::VoxelGrid<PointType> downSizeFilter;
  cv::Mat rangeMat;

  bool odomDeskewFlag;
  float odomIncreX;
  float odomIncreY;
  float odomIncreZ;

  std::shared_ptr<lio_sam::cloud_info> cloud_info_;
  double timeScanCur;
  double timeScanEnd;
  std::vector<smoothness_t> cloudSmoothness;
  float* cloudCurvature;
  int* cloudNeighborPicked;
  int* cloudLabel;

  std_msgs::Header cloudHeader;
};