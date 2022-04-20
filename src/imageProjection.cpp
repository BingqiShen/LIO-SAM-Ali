#include "image_projection.h"
inline float pointDistance(PointType p)
{
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
};

ImageProjection::ImageProjection()
{
  LOG(WARNING) << "here";
  allocateMemory();
  //   resetParameters();

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
}

ImageProjection::~ImageProjection() {}

void ImageProjection::allocateMemory()
{
  LOG(WARNING) << "here";
  tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
  tmpHesaiCloudIn.reset(new pcl::PointCloud<HesaiPointXYZIRT>());
  fullCloud.reset(new pcl::PointCloud<PointType>());
  extractedCloud.reset(new pcl::PointCloud<PointType>());

  fullCloud->points.resize(N_SCAN * Horizon_SCAN);

  cloud_info_.reset(new lio_sam::cloud_info());
  cloud_info_->startRingIndex.assign(N_SCAN, 0);
  cloud_info_->endRingIndex.assign(N_SCAN, 0);

  cloud_info_->pointColInd.assign(N_SCAN * Horizon_SCAN, 0);
  cloud_info_->pointRange.assign(N_SCAN * Horizon_SCAN, 0);

  LOG(WARNING) << cloud_info_->startRingIndex.size();
  cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

  downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize,
                             odometrySurfLeafSize);

  cornerCloud.reset(new pcl::PointCloud<PointType>());
  surfaceCloud.reset(new pcl::PointCloud<PointType>());

  cloudCurvature = new float[N_SCAN * Horizon_SCAN];
  cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
  cloudLabel = new int[N_SCAN * Horizon_SCAN];
  resetParameters();
}

void ImageProjection::resetParameters()
{
  extractedCloud->clear();
  // reset range matrix for range image projection
  rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

  //   cloud_info_->startRingIndex.clear();
  //   cloud_info_->endRingIndex.clear();
  //   cloud_info_->pointColInd.clear();
  //   cloud_info_->pointRange.clear();
}

void ImageProjection::cloudHandler(
    const pcl::PointCloud<PointXYZIRT>::Ptr& cloud,
    std::unordered_map<double, Eigen::Affine3f> points_pose,
    const bool& imu_valid, const bool& odom_valid,const Eigen::Affine3f& lidar2base)
{
  cloud_info_->imuAvailable = imu_valid;
  cloud_info_->odomAvailable = odom_valid;
  projectPointCloud(cloud, points_pose,lidar2base);
  cloudExtraction();
  calculateSmoothness();
  markOccludedPoints();

  extractFeatures();
  //         publishFeatureCloud();
  resetParameters();
}

PointType ImageProjection::deskewPoint(PointType* point,
                                       const Eigen::Affine3f& transBt)
{
  PointType newPoint;
  newPoint.x = transBt(0, 0) * point->x + transBt(0, 1) * point->y +
               transBt(0, 2) * point->z + transBt(0, 3);
  newPoint.y = transBt(1, 0) * point->x + transBt(1, 1) * point->y +
               transBt(1, 2) * point->z + transBt(1, 3);
  newPoint.z = transBt(2, 0) * point->x + transBt(2, 1) * point->y +
               transBt(2, 2) * point->z + transBt(2, 3);
  newPoint.intensity = point->intensity;
  return newPoint;
}

void ImageProjection::projectPointCloud(
    const pcl::PointCloud<PointXYZIRT>::Ptr& laserCloudIn,
    std::unordered_map<double, Eigen::Affine3f> points_pose,
    const Eigen::Affine3f& lidar2base)
{
  int cloudSize = laserCloudIn->points.size();
  //   LOG(WARNING) <<cloudSize;
  // range image projection
  Eigen::Affine3f base2lidar = lidar2base.inverse();
  for (int i = 0; i < cloudSize; ++i)
  {
    PointType thisPoint;
    thisPoint.x = laserCloudIn->points[i].x;
    thisPoint.y = laserCloudIn->points[i].y;
    thisPoint.z = laserCloudIn->points[i].z;
    thisPoint.intensity = laserCloudIn->points[i].intensity;

    float range = pointDistance(thisPoint);
    if (range < lidarMinRange || range > lidarMaxRange)
      continue;

    int rowIdn = laserCloudIn->points[i].ring;
    if (rowIdn < 0 || rowIdn >= N_SCAN)
      continue;

    if (rowIdn % downsampleRate != 0)
      continue;

    float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

    static float ang_res_x = 360.0 / float(Horizon_SCAN);
    int columnIdn =
        -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
    if (columnIdn >= Horizon_SCAN)
      columnIdn -= Horizon_SCAN;

    if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
      continue;

    if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
      continue;
    //
    //     if(i ==0){
    //
    //       LOG(WARNING) <<std::setprecision(16)<<
    //       laserCloudIn->points[i].time<<": " <<
    //       points_pose[laserCloudIn->points[i].time].matrix() ;
    //     }
    thisPoint =
        deskewPoint(&thisPoint, base2lidar * points_pose[laserCloudIn->points[i].time] * lidar2base);
    //     LOG(WARNING) <<"1";
    rangeMat.at<float>(rowIdn, columnIdn) = range;

    int index = columnIdn + rowIdn * Horizon_SCAN;
    fullCloud->points[index] = thisPoint;
  }
}

void ImageProjection::cloudExtraction()
{
  int count = 0;
  // extract segmented cloud for lidar odometry
  for (int i = 0; i < N_SCAN; ++i)
  {
    //     LOG(WARNING) <<cloud_info_->startRingIndex.size();
    cloud_info_->startRingIndex[i] = count - 1 + 5;
    for (int j = 0; j < Horizon_SCAN; ++j)
    {
      if (rangeMat.at<float>(i, j) != FLT_MAX)
      {
        // mark the points' column index for marking occlusion later
        cloud_info_->pointColInd[count] = j;
        // save range info
        cloud_info_->pointRange[count] = rangeMat.at<float>(i, j);
        // save extracted cloud
        extractedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
        // size of extracted cloud
        ++count;
      }
    }
    cloud_info_->endRingIndex[i] = count - 1 - 5;
  }
}

void ImageProjection::calculateSmoothness()
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

void ImageProjection::markOccludedPoints()
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

void ImageProjection::extractFeatures()
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
void ImageProjection::getFeatureCloud(
    const Eigen::Affine3f& lidar2base,
    pcl::PointCloud<PointType>::Ptr& corner_cloud,
    pcl::PointCloud<PointType>::Ptr& surface_cloud,
    std::shared_ptr<lio_sam::cloud_info>& cloud_info)
{
  for (PointType& point : cornerCloud->points)
  {
    Eigen::Vector3f p = lidar2base * Eigen::Vector3f(point.x, point.y, point.z);
    point.x = p.x();
    point.y = p.y();
    point.z = p.z();
  }
  for (PointType& point : surfaceCloud->points)
  {
    Eigen::Vector3f p = lidar2base * Eigen::Vector3f(point.x, point.y, point.z);
    point.x = p.x();
    point.y = p.y();
    point.z = p.z();
  }
  *corner_cloud += *cornerCloud;
  *surface_cloud += *surfaceCloud;
  cloud_info = (cloud_info_);
}