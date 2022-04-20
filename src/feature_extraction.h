#include "utility.h"
#include "lio_sam/cloud_info.h"

class FeatureExtraction : public ParamServer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FeatureExtraction();
  void initializationValue();
  std::shared_ptr<lio_sam::cloud_info>
  laserCloudInfoHandler(std::shared_ptr<lio_sam::cloud_info> msgIn);
  void calculateSmoothness();
  void markOccludedPoints();
  void extractFeatures();
  void freeCloudInfoMemory();
  void publishFeatureCloud();

private:
  pcl::PointCloud<PointType>::Ptr extractedCloud;
  pcl::PointCloud<PointType>::Ptr cornerCloud;
  pcl::PointCloud<PointType>::Ptr surfaceCloud;

  pcl::VoxelGrid<PointType> downSizeFilter;

  std::shared_ptr<lio_sam::cloud_info> cloud_info_;
  std_msgs::Header cloudHeader;

  std::vector<smoothness_t> cloudSmoothness;
  float* cloudCurvature;
  int* cloudNeighborPicked;
  int* cloudLabel;
};