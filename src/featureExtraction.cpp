#include "feature_extraction.h"

FeatureExtraction::FeatureExtraction() { initializationValue(); }

void FeatureExtraction::initializationValue()
{
  cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

  downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize,
                             odometrySurfLeafSize);

  extractedCloud.reset(new pcl::PointCloud<PointType>());
  cornerCloud.reset(new pcl::PointCloud<PointType>());
  surfaceCloud.reset(new pcl::PointCloud<PointType>());

  cloudCurvature = new float[N_SCAN * Horizon_SCAN];
  cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
  cloudLabel = new int[N_SCAN * Horizon_SCAN];
}

std::shared_ptr<lio_sam::cloud_info> FeatureExtraction::laserCloudInfoHandler(
    std::shared_ptr<lio_sam::cloud_info> msgIn)
{
  cloud_info_ = std::move(msgIn);    // new cloud info
  cloudHeader = cloud_info_->header; // new cloud header
  pcl::fromROSMsg(cloud_info_->cloud_deskewed,
                  *extractedCloud); // new cloud for extraction

  calculateSmoothness();

  markOccludedPoints();

  extractFeatures();

  publishFeatureCloud();
  return std::move(cloud_info_);
}

void FeatureExtraction::calculateSmoothness()
{
  int cloudSize = extractedCloud->points.size();
  for (int i = 5; i < cloudSize - 5; i++)
  {
    float diffRange =
        cloud_info_->pointRange[i - 5] + cloud_info_->pointRange[i - 4] +
        cloud_info_->pointRange[i - 3] + cloud_info_->pointRange[i - 2] +
        cloud_info_->pointRange[i - 1] - cloud_info_->pointRange[i] * 10 +
        cloud_info_->pointRange[i + 1] + cloud_info_->pointRange[i + 2] +
        cloud_info_->pointRange[i + 3] + cloud_info_->pointRange[i + 4] +
        cloud_info_->pointRange[i + 5];

    cloudCurvature[i] =
        diffRange * diffRange; // diffX * diffX + diffY * diffY + diffZ * diffZ;

    cloudNeighborPicked[i] = 0;
    cloudLabel[i] = 0;
    // cloudSmoothness for sorting
    cloudSmoothness[i].value = cloudCurvature[i];
    cloudSmoothness[i].ind = i;
  }
}

void FeatureExtraction::markOccludedPoints()
{
  int cloudSize = extractedCloud->points.size();
  // mark occluded points and parallel beam points
  for (int i = 5; i < cloudSize - 6; ++i)
  {
    // occluded points
    float depth1 = cloud_info_->pointRange[i];
    float depth2 = cloud_info_->pointRange[i + 1];
    int columnDiff = std::abs(
        int(cloud_info_->pointColInd[i + 1] - cloud_info_->pointColInd[i]));

    if (columnDiff < 10)
    {
      // 10 pixel diff in range image
      if (depth1 - depth2 > 0.3)
      {
        cloudNeighborPicked[i - 5] = 1;
        cloudNeighborPicked[i - 4] = 1;
        cloudNeighborPicked[i - 3] = 1;
        cloudNeighborPicked[i - 2] = 1;
        cloudNeighborPicked[i - 1] = 1;
        cloudNeighborPicked[i] = 1;
      }
      else if (depth2 - depth1 > 0.3)
      {
        cloudNeighborPicked[i + 1] = 1;
        cloudNeighborPicked[i + 2] = 1;
        cloudNeighborPicked[i + 3] = 1;
        cloudNeighborPicked[i + 4] = 1;
        cloudNeighborPicked[i + 5] = 1;
        cloudNeighborPicked[i + 6] = 1;
      }
    }
    // parallel beam
    float diff1 = std::abs(
        float(cloud_info_->pointRange[i - 1] - cloud_info_->pointRange[i]));
    float diff2 = std::abs(
        float(cloud_info_->pointRange[i + 1] - cloud_info_->pointRange[i]));

    if (diff1 > 0.02 * cloud_info_->pointRange[i] &&
        diff2 > 0.02 * cloud_info_->pointRange[i])
      cloudNeighborPicked[i] = 1;
  }
}

void FeatureExtraction::extractFeatures()
{
  cornerCloud->clear();
  surfaceCloud->clear();

  pcl::PointCloud<PointType>::Ptr surfaceCloudScan(
      new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(
      new pcl::PointCloud<PointType>());

  for (int i = 0; i < N_SCAN; i++)
  {
    surfaceCloudScan->clear();

    for (int j = 0; j < 6; j++)
    {

      int sp = (cloud_info_->startRingIndex[i] * (6 - j) +
                cloud_info_->endRingIndex[i] * j) /
               6;
      int ep = (cloud_info_->startRingIndex[i] * (5 - j) +
                cloud_info_->endRingIndex[i] * (j + 1)) /
                   6 -
               1;

      if (sp >= ep)
        continue;

      std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep,
                by_value());

      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--)
      {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] > edgeThreshold)
        {
          largestPickedNum++;
          if (largestPickedNum <= 20)
          {
            cloudLabel[ind] = 1;
            cornerCloud->push_back(extractedCloud->points[ind]);
          }
          else
          {
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++)
          {
            int columnDiff =
                std::abs(int(cloud_info_->pointColInd[ind + l] -
                             cloud_info_->pointColInd[ind + l - 1]));
            if (columnDiff > 10)
              break;
            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--)
          {
            int columnDiff =
                std::abs(int(cloud_info_->pointColInd[ind + l] -
                             cloud_info_->pointColInd[ind + l + 1]));
            if (columnDiff > 10)
              break;
            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++)
      {
        int ind = cloudSmoothness[k].ind;
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < surfThreshold)
        {

          cloudLabel[ind] = -1;
          cloudNeighborPicked[ind] = 1;

          for (int l = 1; l <= 5; l++)
          {

            int columnDiff =
                std::abs(int(cloud_info_->pointColInd[ind + l] -
                             cloud_info_->pointColInd[ind + l - 1]));
            if (columnDiff > 10)
              break;

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--)
          {

            int columnDiff =
                std::abs(int(cloud_info_->pointColInd[ind + l] -
                             cloud_info_->pointColInd[ind + l + 1]));
            if (columnDiff > 10)
              break;

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      for (int k = sp; k <= ep; k++)
      {
        if (cloudLabel[k] <= 0)
        {
          surfaceCloudScan->push_back(extractedCloud->points[k]);
        }
      }
    }

    surfaceCloudScanDS->clear();
    downSizeFilter.setInputCloud(surfaceCloudScan);
    downSizeFilter.filter(*surfaceCloudScanDS);

    *surfaceCloud += *surfaceCloudScanDS;
  }
}

void FeatureExtraction::freeCloudInfoMemory()
{
  cloud_info_->startRingIndex.clear();
  cloud_info_->endRingIndex.clear();
  cloud_info_->pointColInd.clear();
  cloud_info_->pointRange.clear();
}

void FeatureExtraction::publishFeatureCloud()
{
  // free cloud info memory
  freeCloudInfoMemory();
  // save newly extracted features
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*cornerCloud, cloud_info_->cloud_corner);
  pcl::toROSMsg(*cornerCloud, cloud_info_->cloud_corner);
  //         cloud_info_->cloud_corner  = publishCloud(&pubCornerPoints,
  //         cornerCloud,  cloudHeader.stamp, lidarFrame);
  //         cloud_info_->cloud_surface = publishCloud(&pubSurfacePoints,
  //         surfaceCloud, cloudHeader.stamp, lidarFrame);
  // publish to mapOptimization
  //         pubLaserCloudInfo->publish(cloud_info_);
}
//
//
// int main(int argc, char** argv)
// {
//     ros::init(argc, argv, "lio_sam");
//
//     FeatureExtraction FE;
//
//     ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");
//
//     ros::spin();
//
//     return 0;
// }