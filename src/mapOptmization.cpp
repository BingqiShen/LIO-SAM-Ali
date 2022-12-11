#include "utility.h"
#include "lio_sam/cloud_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <geometry_msgs/PoseStamped.h>
using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity
 * is time stamp)
    */
struct PointXYZIRPYT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16; // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRPYT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time,
                                                                 time))

typedef PointXYZIRPYT PointTypePose;
inline float pointDistance(PointType p)
{
  return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
};

inline float pointDistance(PointType p1, PointType p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) +
              (p1.z - p2.z) * (p1.z - p2.z));
};

inline sensor_msgs::PointCloud2
publishCloud(ros::Publisher* thisPub, pcl::PointCloud<PointType>::Ptr thisCloud,
             ros::Time thisStamp, std::string thisFrame)
{
  sensor_msgs::PointCloud2 tempCloud;
  pcl::toROSMsg(*thisCloud, tempCloud);
  tempCloud.header.stamp = thisStamp;
  tempCloud.header.frame_id = thisFrame;
  if (thisPub->getNumSubscribers() != 0)
    thisPub->publish(tempCloud);
  return tempCloud;
}

inline Rigid3d ToRigid(float* origin_pose)
{
  Eigen::Vector3d t(origin_pose[3], origin_pose[4], origin_pose[5]);
  Eigen::Matrix3d matrix_tmp;
  matrix_tmp = (Eigen::AngleAxisd(origin_pose[2], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(origin_pose[1], Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(origin_pose[0], Eigen::Vector3d::UnitX()))
                   .toRotationMatrix();
  Eigen::Quaterniond q(matrix_tmp);
  return Rigid3d(t, q);
}



class mapOptimization : public ParamServer
{

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  //loop factors (now useless) 
  NonlinearFactorGraph gtSAMgraph;
  Values initialEstimate;
  Values optimizedEstimate;
  ISAM2* isam;
  Values isamCurrentEstimate;
  Eigen::MatrixXd poseCovariance;
  std::deque<nav_msgs::Odometry> gps_queue_;
  ros::Subscriber sub_gps_;
  ros::Subscriber sub_loop_;
  pcl::PointCloud<PointType>::Ptr copy_cloud_key_poses_3D_;
  pcl::PointCloud<PointTypePose>::Ptr copy_cloud_key_poses_6D_;
  pcl::VoxelGrid<PointType> down_size_filter_ICP_;
  std::mutex mtx_loop_info_;
  map<int, int> loop_index_container_; // from new to old
  vector<pair<int, int>> loop_index_queue_;
  vector<gtsam::Pose3> loop_pose_queue_;
  vector<gtsam::noiseModel::Diagonal::shared_ptr> loop_noise_queue_;
  deque<std_msgs::Float64MultiArray> loop_info_vec_;
  
  //useful
  ros::Publisher pub_laser_cloud_surround_;
  ros::Publisher pub_laser_odometry_global_;
  ros::Publisher pub_laser_odometry_incremental_;
  ros::Publisher pub_key_poses_;
  ros::Publisher pub_path_;

  ros::Publisher pub_history_key_frames_;
  ros::Publisher pub_icp_key_frames_;
  ros::Publisher pub_recent_key_frames_;
  ros::Publisher pub_recent_key_frame_;
  ros::Publisher pub_cloud_registered_raw_;
  ros::Publisher pub_loop_constraint_edge_;
  ros::Publisher pub_relative_pose_;
  ros::Publisher pub_scan_matched_points_;
  ros::Publisher pub_pose_;
  ros::Publisher cloud_with_pose_pub_;

  ros::Subscriber sub_cloud_;
  ros::Subscriber sub_saveFullCloud_;
  ros::Subscriber sub_manual_localization_;

  
  lio_sam::cloud_info cloud_info_;

  vector<pcl::PointCloud<PointType>::Ptr> corner_cloud_key_frames_;
  vector<pcl::PointCloud<PointType>::Ptr> surf_cloud_key_frames_;

  pcl::PointCloud<PointType>::Ptr cloud_key_poses_3D_;
  pcl::PointCloud<PointTypePose>::Ptr cloud_key_poses_6D_;

  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_last_; 
  pcl::PointCloud<PointType>::Ptr laser_cloud_surf_last_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_last_DS_; 
  pcl::PointCloud<PointType>::Ptr laser_cloud_surf_last_DS_;

  // add by sbq
  pcl::PointCloud<PointType>::Ptr globalMapCloud_full;
  ros::Time time_laser_stamp_full;
  double time_laser_cur_full;



  pcl::PointCloud<PointType>::Ptr laser_cloud_ori_;
  pcl::PointCloud<PointType>::Ptr coeff_sel_;

  std::vector<PointType> laser_cloud_ori_corner_vec_; // corner point holder for parallel computation
  std::vector<PointType> coeff_sel_corner_vec_;
  std::vector<bool> laser_cloud_ori_corner_flag_;
  std::vector<PointType> laser_cloud_ori_surf_vec_; // surf point holder for parallel computation
  std::vector<PointType> coeff_sel_surf_vec_;
  std::vector<bool> laser_cloud_ori_surf_flag_;

  map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>>
      laser_cloud_map_container_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map_DS_;
  pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map_DS_;

  pcl::KdTreeFLANN<PointType>::Ptr kdtree_corner_from_map_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_surf_from_map_;

  pcl::KdTreeFLANN<PointType>::Ptr kdtree_surrounding_key_poses_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_history_key_poses_;

  pcl::VoxelGrid<PointType> down_size_filter_corner_;
  pcl::VoxelGrid<PointType> down_size_filter_surf_;
  ros::Time time_laser_info_stamp_;
  double time_laser_info_cur_;
  

  float transform_to_be_mapped_[6];

  std::mutex mtx_;

  bool is_degenerate_ = false;
  cv::Mat mat_P_;

  int laser_cloud_corner_from_map_DS_num_ = 0;
  int laser_cloud_surf_from_map_DS_num_ = 0;
  int laser_cloud_corner_last_DS_num_ = 0;
  int laser_cloud_surf_last_DS_num_ = 0;

  bool loop_is_closed_ = false;
  nav_msgs::Path global_path_;

  Eigen::Affine3f trans_point_associate_to_map_;
  Eigen::Affine3f incremental_odometry_affine_front_;
  Eigen::Affine3f incremental_odometry_affine_back_;
  float prior_pose_[6];
  int localization_cloud_size_;
  int key_value_;
  const int local_map_max_frames_num_;
  
  mapOptimization() : local_map_max_frames_num_(10000)
  {
    //useless
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    down_size_filter_ICP_.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize,
                                  mappingSurfLeafSize);
    sub_loop_ = nh.subscribe<std_msgs::Float64MultiArray>(
        "lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler,
        this, ros::TransportHints().tcpNoDelay());
    sub_gps_ = nh.subscribe<nav_msgs::Odometry>(
        gpsTopic, 200, &mapOptimization::gpsHandler, this,
        ros::TransportHints().tcpNoDelay());
    pub_history_key_frames_ = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    pub_icp_key_frames_ = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    pub_loop_constraint_edge_ = nh.advertise<visualization_msgs::MarkerArray>(
        "/lio_sam/mapping/loop_closure_constraints", 1);
  
    //useful
    pub_key_poses_ =
        nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
    pub_laser_cloud_surround_ =
        nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
    pub_laser_odometry_global_ =
        nh.advertise<nav_msgs::Odometry>("lio_sam/mapping/odometry", 1);
    pub_laser_odometry_incremental_ = nh.advertise<nav_msgs::Odometry>(
        "lio_sam/mapping/odometry_incremental", 100);
    pub_path_ = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

    sub_cloud_ = nh.subscribe<lio_sam::cloud_info>(
        "lio_sam/feature/cloud_info", 0,
        &mapOptimization::laserCloudInfoHandler, this,
        ros::TransportHints().tcpNoDelay());
    
    // // add by sbq
    // sub_saveFullCloud_ = nh.subscribe<lio_sam::cloud_info>(
    //     "lio_sam/feature/cloud_info", 0,
    //     &mapOptimization::saveFullPointCloudMap, this,
    //     ros::TransportHints().tcpNoDelay());

    pub_pose_ = nh.advertise<geometry_msgs::PoseStamped>("/lio_sam/matched_pose", 10);
    
   
    pub_recent_key_frames_ =
        nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
    pub_recent_key_frame_ = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/mapping/cloud_registered", 1);
    pub_cloud_registered_raw_ = nh.advertise<sensor_msgs::PointCloud2>(
        "lio_sam/mapping/cloud_registered_raw", 1);
    pub_relative_pose_ =
        nh.advertise<geometry_msgs::Pose>("lio_sam/mapping/relative_pose", 1);
    
    down_size_filter_corner_.setLeafSize(
        mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    down_size_filter_surf_.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize,
                                   mappingSurfLeafSize);
    
    allocateMemory();
    localization_cloud_size_ = 10;
    key_value_ = 0;
  }

  void allocateMemory()
  {
    cloud_key_poses_3D_.reset(new pcl::PointCloud<PointType>());
    cloud_key_poses_6D_.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloud_key_poses_3D_.reset(new pcl::PointCloud<PointType>());
    copy_cloud_key_poses_6D_.reset(new pcl::PointCloud<PointTypePose>());

    kdtree_surrounding_key_poses_.reset(new pcl::KdTreeFLANN<PointType>());
    kdtree_history_key_poses_.reset(new pcl::KdTreeFLANN<PointType>());

    laser_cloud_corner_last_.reset(new pcl::PointCloud<
        PointType>()); // corner feature set from odoOptimization
    laser_cloud_surf_last_.reset(new pcl::PointCloud<
        PointType>()); // surf feature set from odoOptimization
    laser_cloud_corner_last_DS_.reset(new pcl::PointCloud<
        PointType>()); // downsampled corner featuer set from odoOptimization
    laser_cloud_surf_last_DS_.reset(new pcl::PointCloud<
        PointType>()); // downsampled surf featuer set from odoOptimization
    laser_cloud_ori_.reset(new pcl::PointCloud<PointType>());
    coeff_sel_.reset(new pcl::PointCloud<PointType>());

    // add by sbq
    globalMapCloud_full.reset(new pcl::PointCloud<PointType>());

    laser_cloud_ori_corner_vec_.resize(N_SCAN * Horizon_SCAN);
    coeff_sel_corner_vec_.resize(N_SCAN * Horizon_SCAN);
    laser_cloud_ori_corner_flag_.resize(N_SCAN * Horizon_SCAN);
    laser_cloud_ori_surf_vec_.resize(N_SCAN * Horizon_SCAN);
    coeff_sel_surf_vec_.resize(N_SCAN * Horizon_SCAN);
    laser_cloud_ori_surf_flag_.resize(N_SCAN * Horizon_SCAN);

    std::fill(laser_cloud_ori_corner_flag_.begin(), laser_cloud_ori_corner_flag_.end(),
              false);
    std::fill(laser_cloud_ori_surf_flag_.begin(), laser_cloud_ori_surf_flag_.end(),
              false);

    laser_cloud_corner_from_map_.reset(new pcl::PointCloud<PointType>());
    laser_cloud_surf_from_map_.reset(new pcl::PointCloud<PointType>());
    laser_cloud_corner_from_map_DS_.reset(new pcl::PointCloud<PointType>());
    laser_cloud_surf_from_map_DS_.reset(new pcl::PointCloud<PointType>());

    kdtree_corner_from_map_.reset(new pcl::KdTreeFLANN<PointType>());
    kdtree_surf_from_map_.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i)
    {
      transform_to_be_mapped_[i] = 0;
    }

    mat_P_ = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
  }
  
  // input: feature cloud extracted
  void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
  {
    // extract time stamp
    time_laser_info_stamp_ = msgIn->header.stamp;
    time_laser_info_cur_ = msgIn->header.stamp.toSec();

    // extract info and feature cloud
    cloud_info_ = *msgIn;
    pcl::fromROSMsg(msgIn->cloud_corner, *laser_cloud_corner_last_);
    pcl::fromROSMsg(msgIn->cloud_surface, *laser_cloud_surf_last_);

    //             *all_cloud += *laserCloudCornerLast;
    //             *all_cloud += *laserCloudSurfLast;
    std::lock_guard<std::mutex> lock(mtx_);

    static double timeLastProcessing = -1;
    static int lidar_cnt = 0;
    if (time_laser_info_cur_ - timeLastProcessing >= mappingProcessInterval)
    {
      timeLastProcessing = time_laser_info_cur_;
      
      // get initial pose put into transformTobeMapped
      updateInitialGuess();
      
      
      // cout <<"\033[34m" << "prior pose: ";
      // for(int i = 0; i < 6; i++)
      //   cout << transformTobeMapped[i] <<", ";
      // cout <<"\033[37m"<<endl;
      for (int i = 0; i < 6; i++)
        prior_pose_[i] = transform_to_be_mapped_[i];
      extractSurroundingKeyFrames();
      downsampleCurrentScan();

      scan2MapOptimization();
      saveKeyFramesAndFactor();

      saveFullPointCloudMap(msgIn);

      publishOdometry(msgIn);
      publishFrames();
      
      //             LOG(WARNING) <<"here";
      // pub localization submap
      if (lidar_cnt++ < 2)
        return;
      else
        lidar_cnt = 0;
      pcl::PointCloud<PointType>::Ptr surf_cloud(
          new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr corner_cloud(
          new pcl::PointCloud<PointType>());

      pcl::fromROSMsg(msgIn->cloud_surface, *surf_cloud);
      pcl::fromROSMsg(msgIn->cloud_corner, *corner_cloud);

    }
  }
  
  //unuse
  void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
  {
    gps_queue_.push_back(*gpsMsg);
  }

  void pointAssociateToMap(PointType const* const pi, PointType* const po)
  {
    po->x = trans_point_associate_to_map_(0, 0) * pi->x +
            trans_point_associate_to_map_(0, 1) * pi->y +
            trans_point_associate_to_map_(0, 2) * pi->z +
            trans_point_associate_to_map_(0, 3);
    po->y = trans_point_associate_to_map_(1, 0) * pi->x +
            trans_point_associate_to_map_(1, 1) * pi->y +
            trans_point_associate_to_map_(1, 2) * pi->z +
            trans_point_associate_to_map_(1, 3);
    po->z = trans_point_associate_to_map_(2, 0) * pi->x +
            trans_point_associate_to_map_(2, 1) * pi->y +
            trans_point_associate_to_map_(2, 2) * pi->z +
            trans_point_associate_to_map_(2, 3);
    po->intensity = pi->intensity;
  }

  // transform the laser points in lidar coordinate to that in wolrd coordinate
  pcl::PointCloud<PointType>::Ptr
  transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                      PointTypePose* transformIn)
  {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    // transform x,y,z,roll,pitch,yaw to T
    Eigen::Affine3f transCur = pcl::getTransformation(
        transformIn->x, transformIn->y, transformIn->z, transformIn->roll,
        transformIn->pitch, transformIn->yaw);

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
      const auto& pointFrom = cloudIn->points[i];
      cloudOut->points[i].x = transCur(0, 0) * pointFrom.x +
                              transCur(0, 1) * pointFrom.y +
                              transCur(0, 2) * pointFrom.z + transCur(0, 3);
      cloudOut->points[i].y = transCur(1, 0) * pointFrom.x +
                              transCur(1, 1) * pointFrom.y +
                              transCur(1, 2) * pointFrom.z + transCur(1, 3);
      cloudOut->points[i].z = transCur(2, 0) * pointFrom.x +
                              transCur(2, 1) * pointFrom.y +
                              transCur(2, 2) * pointFrom.z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }

  //unuse
  gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
  {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll),
                                            double(thisPoint.pitch),
                                            double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y),
                                      double(thisPoint.z)));
  }

  //unuse
  gtsam::Pose3 trans2gtsamPose(float transformIn[])
  {
    return gtsam::Pose3(
        gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
  }

  Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
  {
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z,
                                  thisPoint.roll, thisPoint.pitch,
                                  thisPoint.yaw);
  }

  Eigen::Affine3f trans2Affine3f(float transformIn[])
  {
    return pcl::getTransformation(transformIn[3], transformIn[4],
                                  transformIn[5], transformIn[0],
                                  transformIn[1], transformIn[2]);
  }

  PointTypePose trans2PointTypePose(float transformIn[])
  {
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
  }

  
  // add by sbq
  void saveFullPointCloudMap(const lio_sam::cloud_infoConstPtr& msgIn)
  {
    lio_sam::cloud_info cloud_full;
    pcl::PointCloud<PointType>::Ptr laser_cloud_corner_full;
    laser_cloud_corner_full.reset(new pcl::PointCloud<PointType>()); 
    pcl::PointCloud<PointType>::Ptr laser_cloud_surf_full;
    laser_cloud_surf_full.reset(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalCornerCloud_full(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalSurfCloud_full(new pcl::PointCloud<PointType>());
    PointTypePose KeyPose6D;
    // cout << "111" << endl;
    KeyPose6D = trans2PointTypePose(transform_to_be_mapped_);
    cout << transform_to_be_mapped_[3] << " " 
         << transform_to_be_mapped_[4] << " " 
         << transform_to_be_mapped_[5] << endl;
    // extract time stamp
    time_laser_stamp_full = msgIn->header.stamp;
    time_laser_cur_full = msgIn->header.stamp.toSec();

    // extract info and feature cloud
    cloud_full = *msgIn;
    pcl::fromROSMsg(msgIn->cloud_corner, *laser_cloud_corner_full);
    pcl::fromROSMsg(msgIn->cloud_surface, *laser_cloud_surf_full);
    // cout << "333" << endl;
    *globalCornerCloud_full = *transformPointCloud(laser_cloud_corner_full, &KeyPose6D);
    *globalSurfCloud_full = *transformPointCloud(laser_cloud_surf_full, &KeyPose6D);
    // cout << "444" << endl << endl;
    

    *globalMapCloud_full += *globalCornerCloud_full;
    *globalMapCloud_full += *globalSurfCloud_full;
    
    // if (savePCD == false)
    //     return;
    
    // cout << "****************************************************" << endl;
    // cout << "Saving map to pcd files ..." << endl;
    // // create directory and remove old files;
    // savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
    // int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
    // unused = system((std::string("mkdir ") + savePCDDirectory).c_str());

    // pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd",
    //                           *globalMapCloud_full);
    // cout << "****************************************************" << endl;
    // cout << "Saving map to pcd files completed" << endl; 
  }



  //unuse
  void visualizeGlobalMapThread()
  {
    ros::Rate rate(2);
    while (ros::ok())
    {
      rate.sleep();
      publishGlobalMap();
    }

    if (savePCD == false)
        return;
    
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files ..." << endl;
    // create directory and remove old files;
    savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
    int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
    unused = system((std::string("mkdir ") + savePCDDirectory).c_str());

    pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd",
                              *globalMapCloud_full);
    cout << "****************************************************" << endl;
    cout << "Saving map to pcd files completed" << endl;

    // if (savePCD == false)
    //   return;

    // cout << "****************************************************" << endl;
    // cout << "Saving map to pcd files ..." << endl;
    // // create directory and remove old files;
    // savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
    // int unused =
    //     system((std::string("exec rm -r ") + savePCDDirectory).c_str());
    // unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
    // // save key frame transformations
    // pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd",
    //                           *cloud_key_poses_3D_);
    // pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd",
    //                           *cloud_key_poses_6D_);
    // // extract global point cloud map
    // pcl::PointCloud<PointType>::Ptr globalCornerCloud(
    //     new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(
    //     new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr globalSurfCloud(
    //     new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(
    //     new pcl::PointCloud<PointType>());
    // pcl::PointCloud<PointType>::Ptr globalMapCloud(
    //     new pcl::PointCloud<PointType>());
    // for (int i = 0; i < (int)cloud_key_poses_3D_->size(); i++)
    // {
    //   *globalCornerCloud += *transformPointCloud(corner_cloud_key_frames_[i],
    //                                              &cloud_key_poses_6D_->points[i]);
    //   *globalSurfCloud += *transformPointCloud(surf_cloud_key_frames_[i],
    //                                            &cloud_key_poses_6D_->points[i]);
    //   cout << "\r" << std::flush << "Processing feature cloud " << i << " of "
    //        << cloud_key_poses_6D_->size() << " ...";
    // }
    // // down-sample and save corner cloud
    // down_size_filter_corner_.setInputCloud(globalCornerCloud);
    // down_size_filter_corner_.filter(*globalCornerCloudDS);
    // pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd",
    //                           *globalCornerCloudDS);
    // // down-sample and save surf cloud
    // down_size_filter_surf_.setInputCloud(globalSurfCloud);
    // down_size_filter_surf_.filter(*globalSurfCloudDS);
    // pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd",
    //                           *globalSurfCloudDS);
    // // down-sample and save global point cloud map
    // *globalMapCloud += *globalCornerCloud;
    // *globalMapCloud += *globalSurfCloud;
    // pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd",
    //                           *globalMapCloud);
    // cout << "****************************************************" << endl;
    // cout << "Saving map to pcd files completed" << endl;
  }

  void publishGlobalMap()
  {
    if (pub_laser_cloud_surround_.getNumSubscribers() == 0)
      return;

    if (cloud_key_poses_3D_->points.empty() == true)
      return;

    //         pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new
    //         pcl::KdTreeFLANN<PointType>());;
    //         pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new
    //         pcl::PointCloud<PointType>());
    //         pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new
    //         pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(
        new pcl::PointCloud<PointType>());
    //
    //         // kd-tree to find near key frames to visualize
    //         std::vector<int> pointSearchIndGlobalMap;
    //         std::vector<float> pointSearchSqDisGlobalMap;
    //         // search near key frames to visualize
    //         mtx.lock();
    //         kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    //         kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(),
    //         globalMapVisualizationSearchRadius, pointSearchIndGlobalMap,
    //         pointSearchSqDisGlobalMap, 0);
    //         mtx.unlock();
    for (unsigned int i = 0; i < cloud_key_poses_3D_->points.size(); ++i)
    {
      int thisKeyInd = round(cloud_key_poses_3D_->points[i].intensity -
                             cloud_key_poses_3D_->points[0].intensity);
      if (thisKeyInd < 0)
      {
        thisKeyInd += local_map_max_frames_num_;
      }
      *globalMapKeyFrames +=
          *transformPointCloud(corner_cloud_key_frames_[thisKeyInd],
                               &cloud_key_poses_6D_->points[thisKeyInd]);
      *globalMapKeyFrames +=
          *transformPointCloud(surf_cloud_key_frames_[thisKeyInd],
                               &cloud_key_poses_6D_->points[thisKeyInd]);
    }
    //         for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
    //             globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    //         // downsample near selected key frames
    //         pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for
    //         global map visualization
    //         downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity,
    //         globalMapVisualizationPoseDensity,
    //         globalMapVisualizationPoseDensity); // for global map
    //         visualization
    //         downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    //         downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    //
    //         // extract visualized and downsampled key frames
    //         for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
    //             if (pointDistance(globalMapKeyPosesDS->points[i],
    //             cloudKeyPoses3D->back()) >
    //             globalMapVisualizationSearchRadius)
    //                 continue;
    //             int thisKeyInd =
    //             round(globalMapKeyPosesDS->points[i].intensity -
    //             cloudKeyPoses3D->points[0].intensity);
    // //             int thisKeyInd =
    // round(cloudKeyPoses3D->points[i].intensity
    // -cloudKeyPoses3D->points[0].intensity);
    //             if(thisKeyInd < 0)
    //             {
    //       //         cout << "\33[34m thisKeyInd: "
    //       <<cloudKeyPoses3D->points[i].intensity << ", "
    //       <<(cloudKeyPoses3D->points[0].intensity)<<", "
    //       <<cornerCloudKeyFrames.size() <<endl;
    //               thisKeyInd += local_map_max_frames_num_;
    //             }
    //             if(thisKeyInd < 0 || thisKeyInd >=
    //             local_map_max_frames_num_){
    //               cout <<"error: " << thisKeyInd <<"," <<
    //               cloudKeyPoses3D->points.size() <<endl;
    // //               continue;
    //             }
    //             *globalMapKeyFrames +=
    //             *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],
    //             &cloudKeyPoses6D->points[thisKeyInd]);
    //             *globalMapKeyFrames +=
    //             *transformPointCloud(surfCloudKeyFrames[thisKeyInd],
    //             &cloudKeyPoses6D->points[thisKeyInd]);
    //         }
    // downsample visualized points
    pcl::VoxelGrid<PointType>
        downSizeFilterGlobalMapKeyFrames; // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(
        globalMapVisualizationLeafSize, globalMapVisualizationLeafSize,
        globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pub_laser_cloud_surround_, globalMapKeyFramesDS,
                 time_laser_info_stamp_, odometryFrame);
  }
  
  //unuse
  void loopClosureThread()
  {
    if (loopClosureEnableFlag == false)
      return;

    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
      rate.sleep();
      performLoopClosure();
      visualizeLoopClosure();
    }
  }
  
  //unuse
  void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
  {
    std::lock_guard<std::mutex> lock(mtx_loop_info_);
    if (loopMsg->data.size() != 2)
      return;

    loop_info_vec_.push_back(*loopMsg);

    while (loop_info_vec_.size() > 5)
      loop_info_vec_.pop_front();
  }

  //unuse
  void performLoopClosure()
  {
    if (cloud_key_poses_3D_->points.empty() == true)
      return;

    mtx_.lock();
    *copy_cloud_key_poses_3D_ = *cloud_key_poses_3D_;
    *copy_cloud_key_poses_6D_ = *cloud_key_poses_6D_;
    mtx_.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
      if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        return;

    // extract cloud
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(
        new pcl::PointCloud<PointType>());
    {
      loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
      loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre,
                            historyKeyframeSearchNum);
      if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        return;
      if (pub_history_key_frames_.getNumSubscribers() != 0)
        publishCloud(&pub_history_key_frames_, prevKeyframeCloud,
                     time_laser_info_stamp_, odometryFrame);
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(
        new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (icp.hasConverged() == false ||
        icp.getFitnessScore() > historyKeyframeFitnessScore)
      return;

    // publish corrected cloud
    if (pub_icp_key_frames_.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr closed_cloud(
          new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud,
                               icp.getFinalTransformation());
      publishCloud(&pub_icp_key_frames_, closed_cloud, time_laser_info_stamp_,
                   odometryFrame);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong =
        pclPointToAffine3f(copy_cloud_key_poses_6D_->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect =
        correctionLidarFrame *
        tWrong; // pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom =
        Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo =
        pclPointTogtsamPose3(copy_cloud_key_poses_6D_->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore,
        noiseScore;
    noiseModel::Diagonal::shared_ptr constraintNoise =
        noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx_.lock();
    loop_index_queue_.push_back(make_pair(loopKeyCur, loopKeyPre));
    loop_pose_queue_.push_back(poseFrom.between(poseTo));
    loop_noise_queue_.push_back(constraintNoise);
    mtx_.unlock();

    // add loop constriant
    loop_index_container_[loopKeyCur] = loopKeyPre;
  }

  //unuse
  bool detectLoopClosureDistance(int* latestID, int* closestID)
  {
    int loopKeyCur = copy_cloud_key_poses_3D_->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loop_index_container_.find(loopKeyCur);
    if (it != loop_index_container_.end())
      return false;

    // find the closest history key frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtree_history_key_poses_->setInputCloud(copy_cloud_key_poses_3D_);
    kdtree_history_key_poses_->radiusSearch(
        copy_cloud_key_poses_3D_->back(), historyKeyframeSearchRadius,
        pointSearchIndLoop, pointSearchSqDisLoop, 0);

    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
      int id = pointSearchIndLoop[i];
      if (abs(copy_cloud_key_poses_6D_->points[id].time - time_laser_info_cur_) >
          historyKeyframeSearchTimeDiff)
      {
        loopKeyPre = id;
        break;
      }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
      return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }

  //unuse
  bool detectLoopClosureExternal(int* latestID, int* closestID)
  {
    // this function is not used yet, please ignore it
    int loopKeyCur = -1;
    int loopKeyPre = -1;

    std::lock_guard<std::mutex> lock(mtx_loop_info_);
    if (loop_info_vec_.empty())
      return false;

    double loopTimeCur = loop_info_vec_.front().data[0];
    double loopTimePre = loop_info_vec_.front().data[1];
    loop_info_vec_.pop_front();

    if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
      return false;

    int cloudSize = copy_cloud_key_poses_6D_->size();
    if (cloudSize < 2)
      return false;

    // latest key
    loopKeyCur = cloudSize - 1;
    for (int i = cloudSize - 1; i >= 0; --i)
    {
      if (copy_cloud_key_poses_6D_->points[i].time >= loopTimeCur)
        loopKeyCur = round(copy_cloud_key_poses_6D_->points[i].intensity);
      else
        break;
    }

    // previous key
    loopKeyPre = 0;
    for (int i = 0; i < cloudSize; ++i)
    {
      if (copy_cloud_key_poses_6D_->points[i].time <= loopTimePre)
        loopKeyPre = round(copy_cloud_key_poses_6D_->points[i].intensity);
      else
        break;
    }

    if (loopKeyCur == loopKeyPre)
      return false;

    auto it = loop_index_container_.find(loopKeyCur);
    if (it != loop_index_container_.end())
      return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
  }

  //unuse
  void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes,
                             const int& key, const int& searchNum)
  {
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloud_key_poses_6D_->size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
      int keyNear = key + i;
      if (keyNear < 0 || keyNear >= cloudSize)
        continue;
      *nearKeyframes +=
          *transformPointCloud(corner_cloud_key_frames_[keyNear],
                               &copy_cloud_key_poses_6D_->points[keyNear]);
      *nearKeyframes +=
          *transformPointCloud(surf_cloud_key_frames_[keyNear],
                               &copy_cloud_key_poses_6D_->points[keyNear]);
    }

    if (nearKeyframes->empty())
      return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(
        new pcl::PointCloud<PointType>());
    down_size_filter_ICP_.setInputCloud(nearKeyframes);
    down_size_filter_ICP_.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
  }

  //unuse
  void visualizeLoopClosure()
  {
    if (loop_index_container_.empty())
      return;

    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = time_laser_info_stamp_;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = time_laser_info_stamp_;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    for (auto it = loop_index_container_.begin(); it != loop_index_container_.end();
         ++it)
    {
      int key_cur = it->first;
      int key_pre = it->second;
      geometry_msgs::Point p;
      p.x = copy_cloud_key_poses_6D_->points[key_cur].x;
      p.y = copy_cloud_key_poses_6D_->points[key_cur].y;
      p.z = copy_cloud_key_poses_6D_->points[key_cur].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      p.x = copy_cloud_key_poses_6D_->points[key_pre].x;
      p.y = copy_cloud_key_poses_6D_->points[key_pre].y;
      p.z = copy_cloud_key_poses_6D_->points[key_pre].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pub_loop_constraint_edge_.publish(markerArray);
  }

  void updateInitialGuess()
  {
    // save current transformation before any processing
    incremental_odometry_affine_front_ = trans2Affine3f(transform_to_be_mapped_);

    static Eigen::Affine3f lastImuTransformation;
    // initialization
    if (cloud_key_poses_3D_->points.empty())
    {
      transform_to_be_mapped_[0] = cloud_info_.imuRollInit;
      transform_to_be_mapped_[1] = cloud_info_.imuPitchInit;
      transform_to_be_mapped_[2] = cloud_info_.imuYawInit;
      LOG(WARNING) << transform_to_be_mapped_[0]  <<" " << transform_to_be_mapped_[1] <<" " <<transform_to_be_mapped_[2] ;
      if (!useImuHeadingInitialization)
        transform_to_be_mapped_[2] = 0;

      lastImuTransformation = pcl::getTransformation(
          0, 0, 0, cloud_info_.imuRollInit, cloud_info_.imuPitchInit,
          cloud_info_.imuYawInit); // save imu before return;
      return;
    }

    // use imu pre-integration estimation for pose guess
    static bool lastImuPreTransAvailable = false;
    static Eigen::Affine3f lastImuPreTransformation;
    if (cloud_info_.odomAvailable == true)
    {
      Eigen::Affine3f transBack = pcl::getTransformation(
          cloud_info_.initialGuessX, cloud_info_.initialGuessY,
          cloud_info_.initialGuessZ, cloud_info_.initialGuessRoll,
          cloud_info_.initialGuessPitch, cloud_info_.initialGuessYaw);
      if (lastImuPreTransAvailable == false)
      {
        lastImuPreTransformation = transBack;
        lastImuPreTransAvailable = true;
      }
      else
      {
        Eigen::Affine3f transIncre =
            lastImuPreTransformation.inverse() * transBack;
        Eigen::Affine3f transTobe = trans2Affine3f(transform_to_be_mapped_);
        Eigen::Affine3f transFinal = transTobe * transIncre;
        pcl::getTranslationAndEulerAngles(
            transFinal, transform_to_be_mapped_[3], transform_to_be_mapped_[4],
            transform_to_be_mapped_[5], transform_to_be_mapped_[0],
            transform_to_be_mapped_[1], transform_to_be_mapped_[2]);

        lastImuPreTransformation = transBack;

        lastImuTransformation = pcl::getTransformation(
            0, 0, 0, cloud_info_.imuRollInit, cloud_info_.imuPitchInit,
            cloud_info_.imuYawInit); // save imu before return;
        return;
      }
    }

    // use imu incremental estimation for pose guess (only rotation)
    if (cloud_info_.imuAvailable == true && 0)
    {
      Eigen::Affine3f transBack =
          pcl::getTransformation(0, 0, 0, cloud_info_.imuRollInit,
                                 cloud_info_.imuPitchInit, cloud_info_.imuYawInit);
      Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

      Eigen::Affine3f transTobe = trans2Affine3f(transform_to_be_mapped_);
      Eigen::Affine3f transFinal = transTobe * transIncre;
      pcl::getTranslationAndEulerAngles(
          transFinal, transform_to_be_mapped_[3], transform_to_be_mapped_[4],
          transform_to_be_mapped_[5], transform_to_be_mapped_[0],
          transform_to_be_mapped_[1], transform_to_be_mapped_[2]);

      lastImuTransformation = pcl::getTransformation(
          0, 0, 0, cloud_info_.imuRollInit, cloud_info_.imuPitchInit,
          cloud_info_.imuYawInit); // save imu before return;
      return;
    }
  }

  //unuse
  void extractForLoopClosure()
  {
    pcl::PointCloud<PointType>::Ptr cloudToExtract(
        new pcl::PointCloud<PointType>());
    int numPoses = cloud_key_poses_3D_->size();
    for (int i = numPoses - 1; i >= 0; --i)
    {
      if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
        cloudToExtract->push_back(cloud_key_poses_3D_->points[i]);
      else
        break;
    }

    extractCloud(cloudToExtract);
  }

  void extractNearby()
  {
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(
        new pcl::PointCloud<PointType>());
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    kdtree_surrounding_key_poses_->setInputCloud(cloud_key_poses_3D_); // create kd-tree
    kdtree_surrounding_key_poses_->radiusSearch(
        cloud_key_poses_3D_->back(), (double)surroundingKeyframeSearchRadius,
        pointSearchInd, pointSearchSqDis);
    for (int i = 0; i < (int)pointSearchInd.size(); ++i)
    {
      int id = pointSearchInd[i];
      surroundingKeyPoses->push_back(cloud_key_poses_3D_->points[id]);
    }
    /*  // not downsample key poses
            downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
            downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

            // also extract some latest key frames in case the robot rotates in
       one position
            int numPoses = cloudKeyPoses3D->size();
            for (int i = numPoses-1; i >= 0; --i)
            {
                if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                    surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
                else
                    break;
            }*/
    extractCloud(surroundingKeyPoses);
  }

  void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
  {
    // fuse the map
    laser_cloud_corner_from_map_->clear();
    laser_cloud_surf_from_map_->clear();
    //         LOG(WARNING) <<cloudToExtract->size();
    for (int i = 0; i < (int)cloudToExtract->size(); ++i)
    {
      int thisKeyInd = round(cloudToExtract->points[i].intensity);
      int firstKeyInd = round(cloud_key_poses_3D_->points[0].intensity);
      int det_index = thisKeyInd - firstKeyInd;

      if (det_index < 0)
        det_index += local_map_max_frames_num_;
      if (det_index < 0 || det_index >= local_map_max_frames_num_)
      {
        cout << "error: " << thisKeyInd << ", " << firstKeyInd << ","
             << det_index << "," << cloud_key_poses_3D_->points.size() << endl;
        continue;
      }
      if (pointDistance(cloudToExtract->points[i], cloud_key_poses_3D_->back()) >
              surroundingKeyframeSearchRadius ||
          fabs(cloudToExtract->points[i].z - cloud_key_poses_3D_->back().z) >
              surroundingKeyframeSearchHeightRange)
        continue;

      //             int thisKeyInd = (int)cloudToExtract->points[i].intensity;
      if (laser_cloud_map_container_.find(thisKeyInd) !=
          laser_cloud_map_container_.end())
      {
        // transformed cloud available
        //               LOG(WARNING) <<"here";
        *laser_cloud_corner_from_map_ += laser_cloud_map_container_[thisKeyInd].first;
        *laser_cloud_surf_from_map_ += laser_cloud_map_container_[thisKeyInd].second;
      }
      else
      {
        // transformed cloud not available
        pcl::PointCloud<PointType> laserCloudCornerTemp =
            *transformPointCloud(corner_cloud_key_frames_[det_index],
                                 &cloud_key_poses_6D_->points[det_index]);
        pcl::PointCloud<PointType> laserCloudSurfTemp =
            *transformPointCloud(surf_cloud_key_frames_[det_index],
                                 &cloud_key_poses_6D_->points[det_index]);
        *laser_cloud_corner_from_map_ += laserCloudCornerTemp;
        *laser_cloud_surf_from_map_ += laserCloudSurfTemp;
        laser_cloud_map_container_[thisKeyInd] =
            make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
      }
    }

    // LOG(WARNING) <<"here";
    // Downsample the surrounding corner key frames (or map)
    down_size_filter_corner_.setInputCloud(laser_cloud_corner_from_map_);
    down_size_filter_corner_.filter(*laser_cloud_corner_from_map_DS_);
    laser_cloud_corner_from_map_DS_num_ = laser_cloud_corner_from_map_DS_->size();
    // Downsample the surrounding surf key frames (or map)
    down_size_filter_surf_.setInputCloud(laser_cloud_surf_from_map_);
    down_size_filter_surf_.filter(*laser_cloud_surf_from_map_DS_);
    laser_cloud_surf_from_map_DS_num_ = laser_cloud_surf_from_map_DS_->size();

    // clear map cache if too large
    //         if (laserCloudMapContainer.size() > 1000)
    //             laserCloudMapContainer.clear();
  }

  void extractSurroundingKeyFrames()
  {
    if (cloud_key_poses_3D_->points.empty() == true)
      return;
    extractNearby();
  }

  void downsampleCurrentScan()
  {
    // Downsample cloud from current scan
    laser_cloud_corner_last_DS_->clear();
    down_size_filter_corner_.setInputCloud(laser_cloud_corner_last_);
    down_size_filter_corner_.filter(*laser_cloud_corner_last_DS_);
    laser_cloud_corner_last_DS_num_ = laser_cloud_corner_last_DS_->size();

    laser_cloud_surf_last_DS_->clear();
    down_size_filter_surf_.setInputCloud(laser_cloud_surf_last_);
    down_size_filter_surf_.filter(*laser_cloud_surf_last_DS_);
    laser_cloud_surf_last_DS_num_ = laser_cloud_surf_last_DS_->size();
  }

  void updatePointAssociateToMap()
  {
    trans_point_associate_to_map_ = trans2Affine3f(transform_to_be_mapped_);
  }

  void cornerOptimization()
  {
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laser_cloud_corner_last_DS_num_; i++)
    {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laser_cloud_corner_last_DS_->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtree_corner_from_map_->nearestKSearch(pointSel, 5, pointSearchInd,
                                          pointSearchSqDis);

      cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
      cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

      if (pointSearchSqDis[4] < 1.0)
      {
        float cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < 5; j++)
        {
          cx += laser_cloud_corner_from_map_DS_->points[pointSearchInd[j]].x;
          cy += laser_cloud_corner_from_map_DS_->points[pointSearchInd[j]].y;
          cz += laser_cloud_corner_from_map_DS_->points[pointSearchInd[j]].z;
        }
        cx /= 5;
        cy /= 5;
        cz /= 5;

        float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
        for (int j = 0; j < 5; j++)
        {
          float ax =
              laser_cloud_corner_from_map_DS_->points[pointSearchInd[j]].x - cx;
          float ay =
              laser_cloud_corner_from_map_DS_->points[pointSearchInd[j]].y - cy;
          float az =
              laser_cloud_corner_from_map_DS_->points[pointSearchInd[j]].z - cz;

          a11 += ax * ax;
          a12 += ax * ay;
          a13 += ax * az;
          a22 += ay * ay;
          a23 += ay * az;
          a33 += az * az;
        }
        a11 /= 5;
        a12 /= 5;
        a13 /= 5;
        a22 /= 5;
        a23 /= 5;
        a33 /= 5;

        matA1.at<float>(0, 0) = a11;
        matA1.at<float>(0, 1) = a12;
        matA1.at<float>(0, 2) = a13;
        matA1.at<float>(1, 0) = a12;
        matA1.at<float>(1, 1) = a22;
        matA1.at<float>(1, 2) = a23;
        matA1.at<float>(2, 0) = a13;
        matA1.at<float>(2, 1) = a23;
        matA1.at<float>(2, 2) = a33;

        cv::eigen(matA1, matD1, matV1);

        if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
        {

          float x0 = pointSel.x;
          float y0 = pointSel.y;
          float z0 = pointSel.z;
          float x1 = cx + 0.1 * matV1.at<float>(0, 0);
          float y1 = cy + 0.1 * matV1.at<float>(0, 1);
          float z1 = cz + 0.1 * matV1.at<float>(0, 2);
          float x2 = cx - 0.1 * matV1.at<float>(0, 0);
          float y2 = cy - 0.1 * matV1.at<float>(0, 1);
          float z2 = cz - 0.1 * matV1.at<float>(0, 2);

          float a012 =
              sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) *
                       ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
                   ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) *
                       ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                   ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) *
                       ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

          float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                           (z1 - z2) * (z1 - z2));

          float la =
              ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) +
               (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) /
              a012 / l12;

          float lb =
              -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) -
                (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
              a012 / l12;

          float lc =
              -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) +
                (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) /
              a012 / l12;

          float ld2 = a012 / l12;

          float s = 1 - 0.9 * fabs(ld2);

          coeff.x = s * la;
          coeff.y = s * lb;
          coeff.z = s * lc;
          coeff.intensity = s * ld2;

          if (s > 0.1)
          {
            laser_cloud_ori_corner_vec_[i] = pointOri;
            coeff_sel_corner_vec_[i] = coeff;
            laser_cloud_ori_corner_flag_[i] = true;
          }
        }
      }
    }
  }

  void surfOptimization()
  {
    updatePointAssociateToMap();

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < laser_cloud_surf_last_DS_num_; i++)
    {
      PointType pointOri, pointSel, coeff;
      std::vector<int> pointSearchInd;
      std::vector<float> pointSearchSqDis;

      pointOri = laser_cloud_surf_last_DS_->points[i];
      pointAssociateToMap(&pointOri, &pointSel);
      kdtree_surf_from_map_->nearestKSearch(pointSel, 5, pointSearchInd,
                                        pointSearchSqDis);

      Eigen::Matrix<float, 5, 3> matA0;
      Eigen::Matrix<float, 5, 1> matB0;
      Eigen::Vector3f matX0;

      matA0.setZero();
      matB0.fill(-1);
      matX0.setZero();

      if (pointSearchSqDis[4] < 1.0)
      {
        for (int j = 0; j < 5; j++)
        {
          matA0(j, 0) = laser_cloud_surf_from_map_DS_->points[pointSearchInd[j]].x;
          matA0(j, 1) = laser_cloud_surf_from_map_DS_->points[pointSearchInd[j]].y;
          matA0(j, 2) = laser_cloud_surf_from_map_DS_->points[pointSearchInd[j]].z;
        }

        matX0 = matA0.colPivHouseholderQr().solve(matB0);

        float pa = matX0(0, 0);
        float pb = matX0(1, 0);
        float pc = matX0(2, 0);
        float pd = 1;

        float ps = sqrt(pa * pa + pb * pb + pc * pc);
        pa /= ps;
        pb /= ps;
        pc /= ps;
        pd /= ps;

        bool planeValid = true;
        for (int j = 0; j < 5; j++)
        {
          if (fabs(pa * laser_cloud_surf_from_map_DS_->points[pointSearchInd[j]].x +
                   pb * laser_cloud_surf_from_map_DS_->points[pointSearchInd[j]].y +
                   pc * laser_cloud_surf_from_map_DS_->points[pointSearchInd[j]].z +
                   pd) > 0.2)
          {
            planeValid = false;
            break;
          }
        }

        if (planeValid)
        {
          float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

          float s = 1 -
                    0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x +
                                                pointSel.y * pointSel.y +
                                                pointSel.z * pointSel.z));

          coeff.x = s * pa;
          coeff.y = s * pb;
          coeff.z = s * pc;
          coeff.intensity = s * pd2;

          if (s > 0.1)
          {
            laser_cloud_ori_surf_vec_[i] = pointOri;
            coeff_sel_surf_vec_[i] = coeff;
            laser_cloud_ori_surf_flag_[i] = true;
          }
        }
      }
    }
  }

  void combineOptimizationCoeffs()
  {
    // combine corner coeffs
    for (int i = 0; i < laser_cloud_corner_last_DS_num_; ++i)
    {
      if (laser_cloud_ori_corner_flag_[i] == true)
      {
        laser_cloud_ori_->push_back(laser_cloud_ori_corner_vec_[i]);
        coeff_sel_->push_back(coeff_sel_corner_vec_[i]);
      }
    }
    // combine surf coeffs
    for (int i = 0; i < laser_cloud_surf_last_DS_num_; ++i)
    {
      if (laser_cloud_ori_surf_flag_[i] == true)
      {
        laser_cloud_ori_->push_back(laser_cloud_ori_surf_vec_[i]);
        coeff_sel_->push_back(coeff_sel_surf_vec_[i]);
      }
    }
    // reset flag for next iteration
    std::fill(laser_cloud_ori_corner_flag_.begin(), laser_cloud_ori_corner_flag_.end(),
              false);
    std::fill(laser_cloud_ori_surf_flag_.begin(), laser_cloud_ori_surf_flag_.end(),
              false);
  }

  bool LMOptimization(int iterCount, float prior_translation_weight,
                      float prior_rotation_weight)
  {
    // This optimization is from the original loam_velodyne by Ji Zhang, need to
    // cope with coordinate transformation
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll

    // lidar -> camera
    float srx = sin(transform_to_be_mapped_[1]);
    float crx = cos(transform_to_be_mapped_[1]);
    float sry = sin(transform_to_be_mapped_[2]);
    float cry = cos(transform_to_be_mapped_[2]);
    float srz = sin(transform_to_be_mapped_[0]);
    float crz = cos(transform_to_be_mapped_[0]);

    int laserCloudSelNum = laser_cloud_ori_->size();
    if (laserCloudSelNum < 50)
    {
      return false;
    }

    cv::Mat matA(laserCloudSelNum + 6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAt(6, laserCloudSelNum + 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum + 6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

    PointType pointOri, coeff;

    for (int i = 0; i < laserCloudSelNum; i++)
    {
      // lidar -> camera
      pointOri.x = laser_cloud_ori_->points[i].y;
      pointOri.y = laser_cloud_ori_->points[i].z;
      pointOri.z = laser_cloud_ori_->points[i].x;
      // lidar -> camera
      coeff.x = coeff_sel_->points[i].y;
      coeff.y = coeff_sel_->points[i].z;
      coeff.z = coeff_sel_->points[i].x;
      coeff.intensity = coeff_sel_->points[i].intensity;
      // in camera
      float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y -
                   srx * sry * pointOri.z) *
                      coeff.x +
                  (-srx * srz * pointOri.x - crz * srx * pointOri.y -
                   crx * pointOri.z) *
                      coeff.y +
                  (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y -
                   cry * srx * pointOri.z) *
                      coeff.z;

      float ary = ((cry * srx * srz - crz * sry) * pointOri.x +
                   (sry * srz + cry * crz * srx) * pointOri.y +
                   crx * cry * pointOri.z) *
                      coeff.x +
                  ((-cry * crz - srx * sry * srz) * pointOri.x +
                   (cry * srz - crz * srx * sry) * pointOri.y -
                   crx * sry * pointOri.z) *
                      coeff.z;

      float arz = ((crz * srx * sry - cry * srz) * pointOri.x +
                   (-cry * crz - srx * sry * srz) * pointOri.y) *
                      coeff.x +
                  (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y +
                  ((sry * srz + cry * crz * srx) * pointOri.x +
                   (crz * sry - cry * srx * srz) * pointOri.y) *
                      coeff.z;
      // lidar -> camera
      matA.at<float>(i, 0) = arz;
      matA.at<float>(i, 1) = arx;
      matA.at<float>(i, 2) = ary;
      matA.at<float>(i, 3) = coeff.z;
      matA.at<float>(i, 4) = coeff.x;
      matA.at<float>(i, 5) = coeff.y;
      matB.at<float>(i, 0) = -coeff.intensity;
    }

    for (int i = 0; i < 6; i++)
    {
      const int idx = laserCloudSelNum + i;
      float prior_weight;
      if (i < 3)
        prior_weight = prior_rotation_weight;
      else
        prior_weight = prior_translation_weight;

      matA.at<float>(idx, i) = prior_weight;
      matB.at<float>(idx, 0) =
          -prior_weight * (transform_to_be_mapped_[i] - prior_pose_[i]);
    }

    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (iterCount == 0)
    {

      cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
      cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

      cv::eigen(matAtA, matE, matV);
      matV.copyTo(matV2);

      is_degenerate_ = false;
      float eignThre[6] = {100, 100, 100, 100, 100, 100};
      for (int i = 5; i >= 0; i--)
      {
        if (matE.at<float>(0, i) < eignThre[i])
        {
          for (int j = 0; j < 6; j++)
          {
            matV2.at<float>(i, j) = 0;
          }
          is_degenerate_ = true;
        }
        else
        {
          break;
        }
      }
      mat_P_ = matV.inv() * matV2;
    }

    if (is_degenerate_)
    {
      cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
      matX.copyTo(matX2);
      matX = mat_P_ * matX2;
    }

    transform_to_be_mapped_[0] += matX.at<float>(0, 0);
    transform_to_be_mapped_[1] += matX.at<float>(1, 0);
    transform_to_be_mapped_[2] += matX.at<float>(2, 0);
    transform_to_be_mapped_[3] += matX.at<float>(3, 0);
    transform_to_be_mapped_[4] += matX.at<float>(4, 0);
    transform_to_be_mapped_[5] += matX.at<float>(5, 0);

    float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
    float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                        pow(matX.at<float>(4, 0) * 100, 2) +
                        pow(matX.at<float>(5, 0) * 100, 2));

    if (deltaR < 0.1 && deltaT < 1.0)
    {
      return true; // converged
    }
    return false; // keep optimizing
  }

  void scan2MapOptimization()
  {
    if (cloud_key_poses_3D_->points.empty())
      return;

    if (laser_cloud_corner_last_DS_num_ > edgeFeatureMinValidNum &&
        laser_cloud_surf_last_DS_num_ > surfFeatureMinValidNum)
    {
      kdtree_corner_from_map_->setInputCloud(laser_cloud_corner_from_map_DS_);
      kdtree_surf_from_map_->setInputCloud(laser_cloud_surf_from_map_DS_);

      float prior_translation_weight = 0.f, prior_rotation_weight = 0.f;
//       laser_cloud_ori_->clear();
//       coeff_sel_->clear();
//       cornerOptimization();
//       surfOptimization();
//       combineOptimizationCoeffs();
//       
//       LMOptimization(0, prior_translation_weight,
//                              prior_rotation_weight) ;
//       for (int try_cnt = 0; try_cnt < 2; try_cnt++)
//       {
        for (int iterCount = 0; iterCount < 30; iterCount++)
        {
          laser_cloud_ori_->clear();
          coeff_sel_->clear();

          cornerOptimization();
          surfOptimization();

          combineOptimizationCoeffs();

          if (LMOptimization(iterCount, prior_translation_weight,
                             prior_rotation_weight) == true)
            break;
        }




      transformUpdate();
    }
    else
    {
      LOG(WARNING) <<
          "Not enough features! Only "<< laser_cloud_corner_from_map_DS_num_  
          <<" edge and " <<laser_cloud_surf_from_map_DS_num_ << " planar features available.";
    }
  }

  void transformUpdate()
  {
    if (cloud_info_.imuAvailable == true)
    {
      if (std::abs(cloud_info_.imuPitchInit) < 1.4)
      {
        double imuWeight = imuRPYWeight; // imu rotation
        tf::Quaternion imuQuaternion;
        tf::Quaternion transformQuaternion;
        double rollMid, pitchMid, yawMid;

        // slerp roll
        transformQuaternion.setRPY(transform_to_be_mapped_[0], 0, 0);
        imuQuaternion.setRPY(cloud_info_.imuRollInit, 0, 0);
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
            .getRPY(rollMid, pitchMid, yawMid);
        transform_to_be_mapped_[0] = rollMid;

        // slerp pitch
        transformQuaternion.setRPY(0, transform_to_be_mapped_[1], 0);
        imuQuaternion.setRPY(0, cloud_info_.imuPitchInit, 0);
        tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
            .getRPY(rollMid, pitchMid, yawMid);
        transform_to_be_mapped_[1] = pitchMid;
      }
    }

    transform_to_be_mapped_[0] =
        constraintTransformation(transform_to_be_mapped_[0], rotation_tollerance);
    transform_to_be_mapped_[1] =
        constraintTransformation(transform_to_be_mapped_[1], rotation_tollerance);
    transform_to_be_mapped_[5] =
        constraintTransformation(transform_to_be_mapped_[5], z_tollerance);

    incremental_odometry_affine_back_ = trans2Affine3f(transform_to_be_mapped_);
  }

  float constraintTransformation(float value, float limit)
  {
    if (value < -limit)
      value = -limit;
    if (value > limit)
      value = limit;

    return value;
  }

  //unuse
  bool saveFrame()
  {
    if (cloud_key_poses_3D_->points.empty())
      return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloud_key_poses_6D_->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(
        transform_to_be_mapped_[3], transform_to_be_mapped_[4], transform_to_be_mapped_[5],
        transform_to_be_mapped_[0], transform_to_be_mapped_[1], transform_to_be_mapped_[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
      return false;

    return true;
  }

  //unuse
  void addOdomFactor()
  {
    if (cloud_key_poses_3D_->points.empty())
    {
      noiseModel::Diagonal::shared_ptr priorNoise =
          noiseModel::Diagonal::Variances(
              (Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8)
                  .finished()); // rad*rad, meter*meter
      gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transform_to_be_mapped_),
                                        priorNoise));
      initialEstimate.insert(0, trans2gtsamPose(transform_to_be_mapped_));
    }
    else
    {
      noiseModel::Diagonal::shared_ptr odometryNoise =
          noiseModel::Diagonal::Variances(
              (Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
      gtsam::Pose3 poseFrom =
          pclPointTogtsamPose3(cloud_key_poses_6D_->points.back());
      gtsam::Pose3 poseTo = trans2gtsamPose(transform_to_be_mapped_);
      gtSAMgraph.add(BetweenFactor<Pose3>(
          cloud_key_poses_3D_->size() - 1, cloud_key_poses_3D_->size(),
          poseFrom.between(poseTo), odometryNoise));
      initialEstimate.insert(cloud_key_poses_3D_->size(), poseTo);
    }
  }

  //unuse
  void addGPSFactor()
  {
    if (gps_queue_.empty())
      return;

    // wait for system initialized and settles down
    if (cloud_key_poses_3D_->points.empty())
      return;
    else
    {
      if (pointDistance(cloud_key_poses_3D_->front(), cloud_key_poses_3D_->back()) <
          5.0)
        return;
    }

    // pose covariance small, no need to correct
    if (poseCovariance(3, 3) < poseCovThreshold &&
        poseCovariance(4, 4) < poseCovThreshold)
      return;

    // last gps position
    static PointType lastGPSPoint;

    while (!gps_queue_.empty())
    {
      if (gps_queue_.front().header.stamp.toSec() < time_laser_info_cur_ - 0.2)
      {
        // message too old
        gps_queue_.pop_front();
      }
      else if (gps_queue_.front().header.stamp.toSec() > time_laser_info_cur_ + 0.2)
      {
        // message too new
        break;
      }
      else
      {
        nav_msgs::Odometry thisGPS = gps_queue_.front();
        gps_queue_.pop_front();

        // GPS too noisy, skip
        float noise_x = thisGPS.pose.covariance[0];
        float noise_y = thisGPS.pose.covariance[7];
        float noise_z = thisGPS.pose.covariance[14];
        if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
          continue;

        float gps_x = thisGPS.pose.pose.position.x;
        float gps_y = thisGPS.pose.pose.position.y;
        float gps_z = thisGPS.pose.pose.position.z;
        if (!useGpsElevation)
        {
          gps_z = transform_to_be_mapped_[5];
          noise_z = 0.01;
        }

        // GPS not properly initialized (0,0,0)
        if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
          continue;

        // Add GPS every a few meters
        PointType curGPSPoint;
        curGPSPoint.x = gps_x;
        curGPSPoint.y = gps_y;
        curGPSPoint.z = gps_z;
        if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
          continue;
        else
          lastGPSPoint = curGPSPoint;

        gtsam::Vector Vector3(3);
        Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
        noiseModel::Diagonal::shared_ptr gps_noise =
            noiseModel::Diagonal::Variances(Vector3);
        gtsam::GPSFactor gps_factor(cloud_key_poses_3D_->size(),
                                    gtsam::Point3(gps_x, gps_y, gps_z),
                                    gps_noise);
        gtSAMgraph.add(gps_factor);

        loop_is_closed_ = true;
        break;
      }
    }
  }

  //unuse
  void addLoopFactor()
  {
    if (loop_index_queue_.empty())
      return;

    for (int i = 0; i < (int)loop_index_queue_.size(); ++i)
    {
      int indexFrom = loop_index_queue_[i].first;
      int indexTo = loop_index_queue_[i].second;
      gtsam::Pose3 poseBetween = loop_pose_queue_[i];
      gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loop_noise_queue_[i];
      gtSAMgraph.add(
          BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loop_index_queue_.clear();
    loop_pose_queue_.clear();
    loop_noise_queue_.clear();
    loop_is_closed_ = true;
  }

  void saveKeyFramesAndFactor()
  {
    if (saveFrame() == false)
      return;

    // odom factor
    //         addOdomFactor();

    //         // gps factor
    //         addGPSFactor();
    //
    //         // loop factor
    //         addLoopFactor();

    // cout << "****************************************************" << endl;
    // gtSAMgraph.print("GTSAM Graph:\n");

    // update iSAM
    //         isam->update(gtSAMgraph, initialEstimate);
    //         isam->update();

    //         if (aLoopIsClosed == true)
    //         {
    //             isam->update();
    //             isam->update();
    //             isam->update();
    //             isam->update();
    //             isam->update();
    //         }
    //
    //         gtSAMgraph.resize(0);
    //         initialEstimate.clear();

    // save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    PointTypePose latestEstimate;

    //         isamCurrentEstimate = isam->calculateEstimate();
    //         latestEstimate =
    //         isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
    // cout << "****************************************************" << endl;
    // isamCurrentEstimate.print("Current estimate: ");
    latestEstimate.yaw = transform_to_be_mapped_[2];
    latestEstimate.pitch = transform_to_be_mapped_[1];
    latestEstimate.roll = transform_to_be_mapped_[0];
    latestEstimate.x = transform_to_be_mapped_[3];
    latestEstimate.y = transform_to_be_mapped_[4];
    latestEstimate.z = transform_to_be_mapped_[5];
    thisPose3D.x = latestEstimate.x;
    thisPose3D.y = latestEstimate.y;
    thisPose3D.z = latestEstimate.z;
    thisPose3D.intensity = key_value_ + 1; // this can be used as index
    cloud_key_poses_3D_->push_back(thisPose3D);

    //         thisPose6D.x = thisPose3D.x;
    //         thisPose6D.y = thisPose3D.y;
    //         thisPose6D.z = thisPose3D.z;
    //         thisPose6D.intensity = thisPose3D.intensity ; // this can be used
    //         as index
    //         thisPose6D.roll  = latestEstimate.rotation().roll();
    //         thisPose6D.pitch = latestEstimate.rotation().pitch();
    //         thisPose6D.yaw   = latestEstimate.rotation().yaw();
    thisPose6D = latestEstimate;
    thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
    thisPose6D.time = time_laser_info_cur_;
    cloud_key_poses_6D_->push_back(thisPose6D);

    // cout << "****************************************************" << endl;
    // cout << "Pose covariance:" << endl;
    // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl <<
    // endl;
    //         poseCovariance =
    //         isam->marginalCovariance(isamCurrentEstimate.size()-1);

    // save updated transform
    //         transformTobeMapped[0] = latestEstimate.rotation().roll();
    //         transformTobeMapped[1] = latestEstimate.rotation().pitch();
    //         transformTobeMapped[2] = latestEstimate.rotation().yaw();
    //         transformTobeMapped[3] = latestEstimate.translation().x();
    //         transformTobeMapped[4] = latestEstimate.translation().y();
    //         transformTobeMapped[5] = latestEstimate.translation().z();

    // save all the received edge and surf points
    pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(
        new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(
        new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laser_cloud_corner_last_DS_, *thisCornerKeyFrame);
    pcl::copyPointCloud(*laser_cloud_surf_last_DS_, *thisSurfKeyFrame);

    // save key frame cloud
    corner_cloud_key_frames_.push_back(thisCornerKeyFrame);
    surf_cloud_key_frames_.push_back(thisSurfKeyFrame);

    // save path for visualization
    updatePath(thisPose6D);

    PointType pose_oldest = cloud_key_poses_3D_->points.front();
    float dist2 =
        (pose_oldest.x - latestEstimate.x) *
            (pose_oldest.x - latestEstimate.x) +
        (pose_oldest.y - latestEstimate.y) *
            (pose_oldest.y - latestEstimate.y) +
        (pose_oldest.z - latestEstimate.z) * (pose_oldest.z - latestEstimate.z);
    //     cout << "dist2: " <<dist2 <<"\t";
    while (dist2 > 1500 * 1500 ||
           corner_cloud_key_frames_.size() >= local_map_max_frames_num_)
    {
      corner_cloud_key_frames_.erase(corner_cloud_key_frames_.begin());
      surf_cloud_key_frames_.erase(surf_cloud_key_frames_.begin());
      //           outlierCloudKeyFrames.erase(outlierCloudKeyFrames.begin());
      int erase_id = cloud_key_poses_3D_->begin()->intensity;
      cloud_key_poses_3D_->erase(cloud_key_poses_3D_->begin());
      cloud_key_poses_6D_->erase(cloud_key_poses_6D_->begin());
      laser_cloud_map_container_.erase(laser_cloud_map_container_.find(erase_id));
      pose_oldest = cloud_key_poses_3D_->points.front();
      dist2 = (pose_oldest.x - latestEstimate.x) *
                  (pose_oldest.x - latestEstimate.x) +
              (pose_oldest.y - latestEstimate.y) *
                  (pose_oldest.y - latestEstimate.y) +
              (pose_oldest.z - latestEstimate.z) *
                  (pose_oldest.z - latestEstimate.z);
    }
    if (key_value_ >= local_map_max_frames_num_ - 1)
      key_value_ = 0;
    else
      key_value_++;
  }

  void correctPoses()
  {
    if (cloud_key_poses_3D_->points.empty())
      return;

    if (loop_is_closed_ == true)
    {
      // clear map cache
      laser_cloud_map_container_.clear();
      // clear path
      global_path_.poses.clear();
      // update key poses
      int numPoses = isamCurrentEstimate.size();
      for (int i = 0; i < numPoses; ++i)
      {
        cloud_key_poses_3D_->points[i].x =
            isamCurrentEstimate.at<Pose3>(i).translation().x();
        cloud_key_poses_3D_->points[i].y =
            isamCurrentEstimate.at<Pose3>(i).translation().y();
        cloud_key_poses_3D_->points[i].z =
            isamCurrentEstimate.at<Pose3>(i).translation().z();

        cloud_key_poses_6D_->points[i].x = cloud_key_poses_3D_->points[i].x;
        cloud_key_poses_6D_->points[i].y = cloud_key_poses_3D_->points[i].y;
        cloud_key_poses_6D_->points[i].z = cloud_key_poses_3D_->points[i].z;
        cloud_key_poses_6D_->points[i].roll =
            isamCurrentEstimate.at<Pose3>(i).rotation().roll();
        cloud_key_poses_6D_->points[i].pitch =
            isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
        cloud_key_poses_6D_->points[i].yaw =
            isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

        updatePath(cloud_key_poses_6D_->points[i]);
      }

      loop_is_closed_ = false;
    }
  }

  void updatePath(const PointTypePose& pose_in)
  {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf::Quaternion q =
        tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    global_path_.poses.push_back(pose_stamped);
  }

  void publishOdometry(const lio_sam::cloud_infoConstPtr& msgIn)
  {
    // Publish odometry for ROS (global)

    nav_msgs::Odometry laserOdometryROS;
    laserOdometryROS.header.stamp = time_laser_info_stamp_;
    laserOdometryROS.header.frame_id = odometryFrame;
    laserOdometryROS.child_frame_id = "odom_mapping";
    laserOdometryROS.pose.pose.position.x = transform_to_be_mapped_[3];
    laserOdometryROS.pose.pose.position.y = transform_to_be_mapped_[4];
    laserOdometryROS.pose.pose.position.z = transform_to_be_mapped_[5];
    laserOdometryROS.pose.pose.orientation =
        tf::createQuaternionMsgFromRollPitchYaw(transform_to_be_mapped_[0],
                                                transform_to_be_mapped_[1],
                                                transform_to_be_mapped_[2]);
    pub_laser_odometry_global_.publish(laserOdometryROS);
    
    
    geometry_msgs::PoseStamped matched_pose;
    matched_pose.header = laserOdometryROS.header;
    matched_pose.pose.position.x = transform_to_be_mapped_[3];
    matched_pose.pose.position.y = transform_to_be_mapped_[4];
    matched_pose.pose.position.z = transform_to_be_mapped_[5];
    
    matched_pose.pose.orientation.w = laserOdometryROS.pose.pose.orientation.w;
    matched_pose.pose.orientation.x = laserOdometryROS.pose.pose.orientation.x;
    matched_pose.pose.orientation.y = laserOdometryROS.pose.pose.orientation.y;
    matched_pose.pose.orientation.z = laserOdometryROS.pose.pose.orientation.z;
    pub_pose_.publish(matched_pose);
          
    //         Publish odometry for ROS (incremental)
    static bool lastIncreOdomPubFlag = false;
    static nav_msgs::Odometry laserOdomIncremental; // incremental odometry msg
    static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
    if (lastIncreOdomPubFlag == false)
    {
      lastIncreOdomPubFlag = true;
      laserOdomIncremental = laserOdometryROS;
      increOdomAffine = trans2Affine3f(transform_to_be_mapped_);
    }
    else
    {
      Eigen::Affine3f affineIncre = incremental_odometry_affine_front_.inverse() *
                                    incremental_odometry_affine_back_;
      increOdomAffine = increOdomAffine * affineIncre;
      float x, y, z, roll, pitch, yaw;
      pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch,
                                        yaw);
      if (cloud_info_.imuAvailable == true)
      {
        if (std::abs(cloud_info_.imuPitchInit) < 1.4)
        {
          double imuWeight = 0.05;
          tf::Quaternion imuQuaternion;
          tf::Quaternion transformQuaternion;
          double rollMid, pitchMid, yawMid;

          // slerp roll
          transformQuaternion.setRPY(roll, 0, 0);
          imuQuaternion.setRPY(cloud_info_.imuRollInit, 0, 0);
          tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
              .getRPY(rollMid, pitchMid, yawMid);
          roll = rollMid;

          // slerp pitch
          transformQuaternion.setRPY(0, pitch, 0);
          imuQuaternion.setRPY(0, cloud_info_.imuPitchInit, 0);
          tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight))
              .getRPY(rollMid, pitchMid, yawMid);
          pitch = pitchMid;
        }
      }
      laserOdomIncremental.header.stamp = time_laser_info_stamp_;
      laserOdomIncremental.header.frame_id = odometryFrame;
      laserOdomIncremental.child_frame_id = "odom_mapping";
      laserOdomIncremental.pose.pose.position.x = x;
      laserOdomIncremental.pose.pose.position.y = y;
      laserOdomIncremental.pose.pose.position.z = z;
      laserOdomIncremental.pose.pose.orientation =
          tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
      if (is_degenerate_)
        laserOdomIncremental.pose.covariance[0] = 1;
      else
        laserOdomIncremental.pose.covariance[0] = 0;
    }
    pub_laser_odometry_incremental_.publish(laserOdomIncremental);
  }

  void publishFrames()
  {
    if (cloud_key_poses_3D_->points.empty())
      return;
    if(debug_)
    {
      // publish key poses
      publishCloud(&pub_key_poses_, cloud_key_poses_3D_, time_laser_info_stamp_,
                  odometryFrame);
      // Publish surrounding key frames
      publishCloud(&pub_recent_key_frames_, laser_cloud_surf_from_map_DS_,
                  time_laser_info_stamp_, odometryFrame);
      // publish registered key frame
      if (pub_recent_key_frame_.getNumSubscribers() != 0)
      {
        pcl::PointCloud<PointType>::Ptr cloudOut(
            new pcl::PointCloud<PointType>());
        PointTypePose thisPose6D = trans2PointTypePose(transform_to_be_mapped_);
        *cloudOut += *transformPointCloud(laser_cloud_corner_last_DS_, &thisPose6D);
        *cloudOut += *transformPointCloud(laser_cloud_surf_last_DS_, &thisPose6D);
        publishCloud(&pub_recent_key_frame_, cloudOut, time_laser_info_stamp_,
                    odometryFrame);
      }
      
    }
    
  }

};

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_INFO);
  FLAGS_colorlogtostderr = true;
  ros::init(argc, argv, "lio_sam");

  mapOptimization MO;

  ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

  std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
  std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread,
                                 &MO);
  

  ros::spin();

  loopthread.join();
  visualizeMapThread.join();

  return 0;
}
