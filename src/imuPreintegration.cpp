#include "utility.h"

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
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
// #include <calib_factor.h>
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

class TransformFusion : public ParamServer
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	std::mutex mtx_;

	ros::Subscriber subImuOdometry;
	ros::Subscriber subLaserOdometry;

	ros::Publisher pub_imu_odometry_;
	ros::Publisher pub_imu_path_;
	ros::Publisher pub_det_imu_path_;

	Eigen::Affine3f lidar_odom_affine_;
	Eigen::Affine3f imu_odom_affine_front_;
	Eigen::Affine3f imu_odom_affine_back_;

	double lidar_odom_time_ = -1;
	deque<nav_msgs::Odometry> imu_odom_queue_;
	Eigen::Vector3d linear_velocity_from_poses_;
	map<double, Rigid3d, std::less<double>, Eigen::aligned_allocator<std::pair<const double, Rigid3d>>> lidar_poses_with_time_;
	bool use_constant_velocity_;
	//   gtsam::Pose3 imu2Base_;
	TransformFusion()
	{
		subLaserOdometry = nh.subscribe<nav_msgs::Odometry>(
		"lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler,
		this, ros::TransportHints().tcpNoDelay());
		subImuOdometry = nh.subscribe<nav_msgs::Odometry>(
		odomTopic + "_incremental", 2000, &TransformFusion::imuOdometryHandler,
		this, ros::TransportHints().tcpNoDelay());

		pub_imu_odometry_ = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
		pub_imu_path_ = nh.advertise<nav_msgs::Path>("lio_sam/imu/path", 1);
		pub_det_imu_path_ = nh.advertise<nav_msgs::Path>("lio_sam/imu_det_path", 1);

		//     imu2Base_ = gtsam::Pose3(
		//           gtsam::Rot3(1,0,0,0),
		//           gtsam::Point3(imu2base_t_.x(), imu2base_t_.y(), imu2base_t_.z()));
		use_constant_velocity_ = false;
	}

	Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
	{
		double x, y, z, roll, pitch, yaw;
		x = odom.pose.pose.position.x;
		y = odom.pose.pose.position.y;
		z = odom.pose.pose.position.z;
		tf::Quaternion orientation;
		tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
		tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
		return pcl::getTransformation(x, y, z, roll, pitch, yaw);
	}

  	void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
  	{
		std::lock_guard<std::mutex> lock(mtx_);

		lidar_odom_affine_ = odom2affine(*odomMsg);

		lidar_odom_time_ = odomMsg->header.stamp.toSec();
		double currentCorrectionTime = ROS_TIME(odomMsg);

		Eigen::Vector3d t(odomMsg->pose.pose.position.x,
				odomMsg->pose.pose.position.y,
				odomMsg->pose.pose.position.z);

		Eigen::Quaterniond q(
			odomMsg->pose.pose.orientation.w, odomMsg->pose.pose.orientation.x,
			odomMsg->pose.pose.orientation.y, odomMsg->pose.pose.orientation.z);

		auto temp = Rigid3d(t, q);
		lidar_poses_with_time_[currentCorrectionTime] = temp;
		if (lidar_poses_with_time_.size() > 2)
			lidar_poses_with_time_.erase(lidar_poses_with_time_.begin());
		
  	}

	// odomTopic+"_incremental
	void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
	{
		std::lock_guard<std::mutex> lock(mtx_);

		imu_odom_queue_.push_back(*odomMsg);
		double imuTime = imu_odom_queue_.back().header.stamp.toSec();
		// get latest odometry (at current IMU stamp)
		if (lidar_odom_time_ == -1)
			return;

		// prevent the program from crashing when imu data come late.
		if (imuTime < lidar_odom_time_)
		{
			LOG(WARNING) << std::setprecision(16) << imuTime << "(imu) should be bigger than (lidarOdom)" << lidar_odom_time_;
			return;
		}

		while (!imu_odom_queue_.empty())
		{
			if (imu_odom_queue_.front().header.stamp.toSec() <= lidar_odom_time_)
				imu_odom_queue_.pop_front();
			else
				break;
		}

		Eigen::Affine3f imuOdomAffineFront = odom2affine(imu_odom_queue_.front());
		Eigen::Affine3f imuOdomAffineBack = odom2affine(imu_odom_queue_.back());
		Eigen::Affine3f imuOdomAffineIncre =
			imuOdomAffineFront.inverse() * imuOdomAffineBack;
		Eigen::Affine3f imuOdomAffineLast = lidar_odom_affine_ * imuOdomAffineIncre;
		float x, y, z, roll, pitch, yaw;
		pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch,
						yaw);
		if (lidar_poses_with_time_.size() > 1 && use_constant_velocity_)
		{
			linear_velocity_from_poses_ =
				(lidar_poses_with_time_.rbegin()->second.translation() -
				lidar_poses_with_time_.begin()->second.translation()) /
				(lidar_poses_with_time_.rbegin()->first -
				lidar_poses_with_time_.begin()->first);
			Eigen::Vector3d cur_translation =
				lidar_poses_with_time_.rbegin()->second.translation() +
				linear_velocity_from_poses_ *
				(imuTime - lidar_poses_with_time_.rbegin()->first);
			x = cur_translation[0];
			y = cur_translation[1];
			z = cur_translation[2];
		}
		// publish latest odometry
		nav_msgs::Odometry laserOdometry = imu_odom_queue_.back();
		laserOdometry.pose.pose.position.x = x;
		laserOdometry.pose.pose.position.y = y;
		laserOdometry.pose.pose.position.z = z;
		laserOdometry.pose.pose.orientation =
			tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
		pub_imu_odometry_.publish(laserOdometry);

		// publish tf
		static tf::TransformBroadcaster tfOdom2BaseLink;
		tf::Transform tCur;
		tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
		tf::StampedTransform odom_2_baselink = tf::StampedTransform(
			tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
		if (vtr_mode_ == 1)
			tfOdom2BaseLink.sendTransform(odom_2_baselink);

		if (debug_)
		{
		// publish IMU path
		static nav_msgs::Path imuPath;
		static double last_path_time = -1;
		static nav_msgs::Path imu_path;
		float p_x = odomMsg->pose.pose.position.x;
		float p_y = odomMsg->pose.pose.position.y;
		float p_z = odomMsg->pose.pose.position.z;
		float r_x = odomMsg->pose.pose.orientation.x;
		float r_y = odomMsg->pose.pose.orientation.y;
		float r_z = odomMsg->pose.pose.orientation.z;
		float r_w = odomMsg->pose.pose.orientation.w;
		bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
		gtsam::Pose3 lidarPose =
			gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z),
				gtsam::Point3(p_x, p_y, p_z));

		//       gtsam::Pose3 imu_pose = lidarPose.compose(imu2Base_);

		geometry_msgs::PoseStamped pose_stamped;
		pose_stamped.header.stamp = imu_odom_queue_.back().header.stamp;
		pose_stamped.header.frame_id = "odom";
		pose_stamped.pose.position.x = lidarPose.x();
		pose_stamped.pose.position.y = lidarPose.y();
		pose_stamped.pose.position.z = lidarPose.z();
		imu_path.poses.push_back(pose_stamped);
		while (!imu_path.poses.empty() &&
			imu_path.poses.front().header.stamp.toSec() < lidar_odom_time_ - 1.0)
			imu_path.poses.erase(imu_path.poses.begin());
		if (pub_det_imu_path_.getNumSubscribers() != 0)
		{
			imu_path.header.stamp = imu_odom_queue_.back().header.stamp;
			imu_path.header.frame_id = "odom";
			pub_det_imu_path_.publish(imu_path);
		}

		if (imuTime - last_path_time > 0.1)
		{
			last_path_time = imuTime;
			geometry_msgs::PoseStamped pose_stamped;
			pose_stamped.header.stamp = imu_odom_queue_.back().header.stamp;
			pose_stamped.header.frame_id = odometryFrame;
			pose_stamped.pose = laserOdometry.pose.pose;
			imuPath.poses.push_back(pose_stamped);
			while (!imuPath.poses.empty() &&
			imuPath.poses.front().header.stamp.toSec() < lidar_odom_time_ - 1.0)
			imuPath.poses.erase(imuPath.poses.begin());
			if (pub_imu_path_.getNumSubscribers() != 0)
			{
			imuPath.header.stamp = imu_odom_queue_.back().header.stamp;
			imuPath.header.frame_id = odometryFrame;
			pub_imu_path_.publish(imuPath);
			}
		}
		}
		}
	};

class IMUPreintegration : public ParamServer
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	std::mutex mtx_;

	ros::Publisher pub_imu_odometry_;
	ros::Publisher bias_pub_;
	ros::Publisher pub_path_;
	ros::Publisher pub_pose_;

	ros::Subscriber sub_imu_;
	ros::Subscriber sub_opt_odometry_;
	ros::Subscriber sub_wheel_odom_;

	gtsam::noiseModel::Diagonal::shared_ptr prior_pose_noise_;
	gtsam::noiseModel::Diagonal::shared_ptr prior_vel_noise_;
	gtsam::noiseModel::Diagonal::shared_ptr prior_bias_noise_;
	gtsam::noiseModel::Diagonal::shared_ptr correction_noise_;
	gtsam::noiseModel::Diagonal::shared_ptr correction_noise2_;
	gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
	gtsam::Vector noise_model_between_bias_;

	gtsam::PreintegratedImuMeasurements *imu_integrator_opt_;
	gtsam::PreintegratedImuMeasurements *imu_integrator_imu_;

	gtsam::Vector3 prev_vel_;
	gtsam::NavState prev_state_;
	gtsam::imuBias::ConstantBias prev_bias_;

	gtsam::NavState prev_state_odom_;
	gtsam::imuBias::ConstantBias prev_bias_odom_;
	gtsam::Pose3 prev_pose_imu_in_local_;
	gtsam::Pose3 imu2Base_, base2Imu_;

	gtsam::ISAM2 optimizer_;
	gtsam::NonlinearFactorGraph graph_factors_;
	gtsam::Values graph_values_;

	std::deque<sensor_msgs::Imu> imu_que_opt_;
	std::deque<sensor_msgs::Imu> imu_que_imu_;
	std::deque<nav_msgs::Odometry> wheel_odom_msg_queue_;
	vector<Rigid3d> lidar_poses_;
	vector<Rigid3d> imu_poses_;

	std::shared_ptr<ImuTracker> imu_tracker_;

	bool system_initialized_;
	bool done_first_opt_;
	bool initialised_;
	bool use_wheel_odom_;

	double last_imu_time_imu_;
	double last_Imu_time_opt_;
	const double delta_t_;
	int key_;

	Rigid3d last_wheel_odom_pose_;
	nav_msgs::Path global_path_;

  	IMUPreintegration() : system_initialized_(false), done_first_opt_(false), initialised_(false),
                        use_wheel_odom_(false), last_imu_time_imu_(-1), last_Imu_time_opt_(-1), delta_t_(0), key_(1)
  	{
		//     Eigen::Quaterniond base2imu_q = imu2base_q_.conjugate();
		//     Eigen::Vector3d base2imu_t = -(base2imu_q  * imu2base_t_);
		imu2Base_ = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
					gtsam::Point3(imu2base_t_.x(), imu2base_t_.y(), imu2base_t_.z()));
		base2Imu_ = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0),
					gtsam::Point3(-imu2base_t_.x(), -imu2base_t_.y(), -imu2base_t_.z()));
		LOG(WARNING) << "imu2Base_: " << imu2Base_;
		LOG(WARNING) << "base2Imu_: " << base2Imu_;

		//     pub_pose_ =
		//         nh.advertise<geometry_msgs::PoseStamped>("/lio_sam/matched_pose", 10);
		bias_pub_ =
		nh.advertise<geometry_msgs::PoseStamped>("/lio_sam/bias", 10);
		pub_imu_odometry_ =
		nh.advertise<nav_msgs::Odometry>(odomTopic + "_incremental", 2000);
		pub_path_ = nh.advertise<nav_msgs::Path>("/lio_sam/optimized_pose", 1);

		sub_imu_ = nh.subscribe<sensor_msgs::Imu>(
		imuTopic, 2000, &IMUPreintegration::imuHandler, this,
		ros::TransportHints().tcpNoDelay());
		// odometry_incremental should be similar to odometry since no loop used
		sub_opt_odometry_ = nh.subscribe<nav_msgs::Odometry>(
		"lio_sam/mapping/odometry_incremental", 0,
		&IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());
		sub_wheel_odom_ = nh.subscribe<nav_msgs::Odometry>(
		"/sdpx_emma_odom", 5, &IMUPreintegration::wheelOdometryHandler, this,
		ros::TransportHints().tcpNoDelay()); // use postion & oritation


		boost::shared_ptr<gtsam::PreintegrationParams> p =
		gtsam::PreintegrationParams::MakeSharedU(imuGravity);
		//         p->setAccelerometerCovariance(gtsam::I_3x3 * 0.1);
		//         p->setGyroscopeCovariance(gtsam::I_3x3 * 0.1);
		//         p->setIntegrationCovariance(gtsam::I_3x3 * 0.1);
		p->setUse2ndOrderCoriolis(false);
		p->setOmegaCoriolis(gtsam::Vector3(0, 0, 0));
		p->accelerometerCovariance =
		gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2); // acc white noise in continuous
		p->gyroscopeCovariance =
		gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
		p->integrationCovariance =
		gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2); // error committed in integrating position from velocities
		gtsam::imuBias::ConstantBias prior_imu_bias(
		(gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
		// assume zero initial bias

		prior_pose_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
		(gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
		prior_vel_noise_ = gtsam::noiseModel::Isotropic::Sigma(3, 1e4);           // m/s
		prior_bias_noise_ = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);         // 1e-2 ~ 1e-3 seems to be good
		correction_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
		(gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.01, 0.01, 0.01).finished()); // rad,rad,rad,m, m, m
		correction_noise2_ = gtsam::noiseModel::Diagonal::Sigmas(
		(gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m

		odom_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
		(gtsam::Vector(6) << 1e1, 1e1, 1e1, 1e-1, 1e-1, 1e-1).finished());
		noise_model_between_bias_ =
		(gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN,
			imuGyrBiasN, imuGyrBiasN)
			.finished();

		imu_integrator_imu_ = new gtsam::PreintegratedImuMeasurements(
		p, prior_imu_bias); // setting up the IMU integration for IMU message thread
		imu_integrator_opt_ = new gtsam::PreintegratedImuMeasurements(
		p, prior_imu_bias); // setting up the IMU integration for optimization
		initialised_ = false;

	}

	void resetOptimization()
	{
		gtsam::ISAM2Params optParameters;
		optParameters.relinearizeThreshold = 0.1;
		optParameters.relinearizeSkip = 1;
		optimizer_ = gtsam::ISAM2(optParameters);

		gtsam::NonlinearFactorGraph newGraphFactors;
		graph_factors_ = newGraphFactors;

		gtsam::Values NewGraphValues;
		graph_values_ = NewGraphValues;
	}

	void resetParams()
	{
		last_imu_time_imu_ = -1;
		done_first_opt_ = false;
		system_initialized_ = false;
	}

	TimestampedTransform<double> ToTransform(const nav_msgs::Odometry &odomMsg)
	{
		TimestampedTransform<double> tmp;
		tmp.time = odomMsg.header.stamp.toSec();
		Eigen::Vector3d t(odomMsg.pose.pose.position.x,
				odomMsg.pose.pose.position.y,
				odomMsg.pose.pose.position.z);
		Eigen::Quaterniond q(
			odomMsg.pose.pose.orientation.w, odomMsg.pose.pose.orientation.x,
			odomMsg.pose.pose.orientation.y, odomMsg.pose.pose.orientation.z);
		tmp.transform = Rigid3d(t, q);
		return tmp;
	}

	void wheelOdometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
	{
		std::lock_guard<std::mutex> lock(mtx_);
		use_wheel_odom_ = true;
		wheel_odom_msg_queue_.push_back(*odomMsg);
	}

  void odometryHandler(const nav_msgs::Odometry::ConstPtr &odomMsg)
  {
    std::lock_guard<std::mutex> lock(mtx_);

    // make sure we have imu data to integrate
    if (imu_que_opt_.empty())
      return;
    double current_correction_time = ROS_TIME(odomMsg);
    float p_x = odomMsg->pose.pose.position.x;
    float p_y = odomMsg->pose.pose.position.y;
    float p_z = odomMsg->pose.pose.position.z;
    float r_x = odomMsg->pose.pose.orientation.x;
    float r_y = odomMsg->pose.pose.orientation.y;
    float r_z = odomMsg->pose.pose.orientation.z;
    float r_w = odomMsg->pose.pose.orientation.w;
    bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
    gtsam::Pose3 lidarPose =
        gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z),
                     gtsam::Point3(p_x, p_y, p_z));

    static gtsam::Pose3 last_lidar_pose = lidarPose.inverse();

    // 0. initialize system
    if (system_initialized_ == false)
    {
      resetOptimization();

      // pop old IMU message
      while (!imu_que_opt_.empty())
      {
        if (ROS_TIME(&imu_que_opt_.front()) < current_correction_time - delta_t_)
        {
          last_Imu_time_opt_ = ROS_TIME(&imu_que_opt_.front());
          imu_que_opt_.pop_front();
        }
        else
          break;
      }

      nav_msgs::Odometry temp_odom_pose;
      if (!wheel_odom_msg_queue_.empty() &&
          wheel_odom_msg_queue_.front().header.stamp.toSec() >= current_correction_time)
      {
        temp_odom_pose = wheel_odom_msg_queue_.front();
      }
      while (!wheel_odom_msg_queue_.empty())
      {
        //               temp_odom_pose.header.stamp.toSec()
        if (wheel_odom_msg_queue_.front().header.stamp.toSec() >=
            current_correction_time)
        {
          auto start_transfrom = ToTransform(temp_odom_pose);
          auto end_transfrom = ToTransform(wheel_odom_msg_queue_.front());
          last_wheel_odom_pose_ = Interpolate<double>(
              start_transfrom, end_transfrom, current_correction_time);
          break;
        }
        else
          temp_odom_pose = wheel_odom_msg_queue_.front();
        wheel_odom_msg_queue_.pop_front();
      }
      if (wheel_odom_msg_queue_.empty())
      {
        auto start_transfrom = ToTransform(temp_odom_pose);
        last_wheel_odom_pose_ = start_transfrom.transform;
      }

      // initial pose
      prev_pose_imu_in_local_ = lidarPose.compose(imu2Base_);
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prev_pose_imu_in_local_,
                                                 prior_pose_noise_);
      graph_factors_.add(priorPose);
      // initial velocity
      prev_vel_ = gtsam::Vector3(0, 0, 0);
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prev_vel_,
                                                  prior_vel_noise_);
      graph_factors_.add(priorVel);
      // initial bias
      prev_bias_ = gtsam::imuBias::ConstantBias();
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(
          B(0), prev_bias_, prior_bias_noise_);
      graph_factors_.add(priorBias);
      // add values
      graph_values_.insert(X(0), prev_pose_imu_in_local_);
      graph_values_.insert(V(0), prev_vel_);
      graph_values_.insert(B(0), prev_bias_);
      // optimize once
      optimizer_.update(graph_factors_, graph_values_);
      graph_factors_.resize(0);
      graph_values_.clear();

      imu_integrator_imu_->resetIntegrationAndSetBias(prev_bias_);
      imu_integrator_opt_->resetIntegrationAndSetBias(prev_bias_);

      key_ = 1;
      system_initialized_ = true;
      return;
    }

    // reset graph for speed
    if (key_ == 100)
    {
      // get updated noise before reset
      gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise =
          gtsam::noiseModel::Gaussian::Covariance(
              optimizer_.marginalCovariance(X(key_ - 1)));
      gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise =
          gtsam::noiseModel::Gaussian::Covariance(
              optimizer_.marginalCovariance(V(key_ - 1)));
      gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise =
          gtsam::noiseModel::Gaussian::Covariance(
              optimizer_.marginalCovariance(B(key_ - 1)));
      // reset graph
      resetOptimization();
      // add pose
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prev_pose_imu_in_local_,
                                                 updatedPoseNoise);
      graph_factors_.add(priorPose); // prior pose factor
      // add velocity
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prev_vel_,
                                                  updatedVelNoise);
      graph_factors_.add(priorVel); // prior vel factor
      // add bias
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(
          B(0), prev_bias_, updatedBiasNoise);
      graph_factors_.add(priorBias); // prior bias factor
      // add values
      graph_values_.insert(X(0), prev_pose_imu_in_local_);
      graph_values_.insert(V(0), prev_vel_);
      graph_values_.insert(B(0), prev_bias_);
      // optimize once
      optimizer_.update(graph_factors_, graph_values_);
      graph_factors_.resize(0);
      graph_values_.clear();
      //             imu_integrator_imu_->resetIntegrationAndSetBias(prev_bias_);
      //             imu_integrator_opt_->resetIntegrationAndSetBias(prev_bias_);
      key_ = 1;
    }

    if (key_ == 1000)
    {
      resetOptimization();
      // add pose
      gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prev_pose_imu_in_local_,
                                                 prior_pose_noise_);
      graph_factors_.add(priorPose); // prior pose factor
      // add velocity
      gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prev_vel_,
                                                  prior_vel_noise_);
      graph_factors_.add(priorVel); // prior vel factor
      // add bias
      gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(
          B(0), prev_bias_, prior_bias_noise_);
      graph_factors_.add(priorBias); // prior bias factor
      // add values
      graph_values_.insert(X(0), prev_pose_imu_in_local_);
      graph_values_.insert(V(0), prev_vel_);
      graph_values_.insert(B(0), prev_bias_);
      // optimize once
      optimizer_.update(graph_factors_, graph_values_);
      graph_factors_.resize(0);
      graph_values_.clear();
      imu_integrator_imu_->resetIntegrationAndSetBias(prev_bias_);
      imu_integrator_opt_->resetIntegrationAndSetBias(prev_bias_);
      key_ = 1;
    }

    // 1. integrate imu data and optimize
    while (!imu_que_opt_.empty())
    {
      // pop and integrate imu data that is between two optimizations
      sensor_msgs::Imu *thisImu = &imu_que_opt_.front();
      double imuTime = ROS_TIME(thisImu);
      if (imuTime < current_correction_time - delta_t_)
      {
        double dt =
            (last_Imu_time_opt_ < 0) ? (1.0 / 500.0) : (imuTime - last_Imu_time_opt_);
        imu_integrator_opt_->integrateMeasurement(
            gtsam::Vector3(thisImu->linear_acceleration.x,
                           thisImu->linear_acceleration.y,
                           thisImu->linear_acceleration.z),
            gtsam::Vector3(thisImu->angular_velocity.x,
                           thisImu->angular_velocity.y,
                           thisImu->angular_velocity.z),
            dt);

        last_Imu_time_opt_ = imuTime;
        imu_que_opt_.pop_front();
      }
      else
        break;
    }
    if (use_wheel_odom_ && 0)
    {
      Rigid3d current_odom_pose;
      nav_msgs::Odometry temp_odom_msg;
      while (!wheel_odom_msg_queue_.empty())
      {
        if (wheel_odom_msg_queue_.front().header.stamp.toSec() >=
            current_correction_time)
        {
          auto start_transfrom = ToTransform(temp_odom_msg);
          auto end_transfrom = ToTransform(wheel_odom_msg_queue_.front());
          current_odom_pose = Interpolate<double>(
              start_transfrom, end_transfrom, current_correction_time);
          break;
        }
        else
          temp_odom_msg = wheel_odom_msg_queue_.front();
        wheel_odom_msg_queue_.pop_front();
      }
      if (wheel_odom_msg_queue_.empty())
      {
        Eigen::Vector3d linear_velocity(temp_odom_msg.twist.twist.linear.x,
                                        temp_odom_msg.twist.twist.linear.y,
                                        temp_odom_msg.twist.twist.linear.z);
        Eigen::Vector3d angular_velocity(temp_odom_msg.twist.twist.angular.x,
                                         temp_odom_msg.twist.twist.angular.y,
                                         temp_odom_msg.twist.twist.angular.z);

        auto start_transfrom = ToTransform(temp_odom_msg);
        double det_t = (current_correction_time - start_transfrom.time);
        Eigen::Vector3d cur_t =
            start_transfrom.transform.translation() + det_t * linear_velocity;

        Eigen::Vector3d det_eul = det_t * angular_velocity;
        Eigen::Matrix3d matrix_tmp;
        matrix_tmp = (Eigen::AngleAxisd(det_eul[2], Eigen::Vector3d::UnitZ()) *
                      Eigen::AngleAxisd(det_eul[1], Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(det_eul[0], Eigen::Vector3d::UnitX()))
                         .toRotationMatrix();
        Eigen::Quaterniond det_q(matrix_tmp);
        current_odom_pose =
            Rigid3d(cur_t, start_transfrom.transform.rotation() * det_q);
      }
      static double last_time = current_correction_time;
      LOG(WARNING) << "current_odom_pose:"
                   << (current_correction_time - last_time) << ", "
                   << current_odom_pose;
      last_time = current_correction_time;
      //         Rigid3d imu2base_pose(imu2base_t_,imu2base_q_);
      Rigid3d det_odom_pose =
          /*imu2base_pose.inverse() **/ last_wheel_odom_pose_.inverse() *
          current_odom_pose;
      last_wheel_odom_pose_ = current_odom_pose;

      // add odom factor
      gtsam::Pose3 wheel_odom_pose =
          gtsam::Pose3(gtsam::Rot3::Quaternion(det_odom_pose.rotation().w(),
                                               det_odom_pose.rotation().x(),
                                               det_odom_pose.rotation().y(),
                                               det_odom_pose.rotation().z()),
                       gtsam::Point3(det_odom_pose.translation().x(),
                                     det_odom_pose.translation().y(),
                                     det_odom_pose.translation().z()));

      gtsam::Pose3 odom_pose_in_imu =
          (base2Imu_.compose(wheel_odom_pose)).compose(imu2Base_);
      //       LOG(WARNING) << odom_pose_in_imu.x() << ", " << odom_pose_in_imu.y()
      //                    << ", " << odom_pose_in_imu.z();

      //         LOG(WARNING) << .x();
      gtsam::Pose3 det_pose =
          (base2Imu_.compose(last_lidar_pose.compose(lidarPose)))
              .compose(imu2Base_);
      //       LOG(WARNING) << det_pose.x() << ", " << det_pose.y() << ", "
      //                    << det_pose.z();
      last_lidar_pose = lidarPose.inverse();
      graph_factors_.add(gtsam::BetweenFactor<gtsam::Pose3>(
          X(key_ - 1), X(key_), wheel_odom_pose,
          odom_noise_)); // odom relative pose factor
    }
    // add imu factor to graph
    const gtsam::PreintegratedImuMeasurements &preint_imu =
        dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(
            *imu_integrator_opt_);
    gtsam::ImuFactor imu_factor(X(key_ - 1), V(key_ - 1), X(key_), V(key_),
                                B(key_ - 1), preint_imu);
    graph_factors_.add(imu_factor);
    // add imu bias between factor
    graph_factors_.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
        B(key_ - 1), B(key_), gtsam::imuBias::ConstantBias(),
        gtsam::noiseModel::Diagonal::Sigmas(
            sqrt(imu_integrator_opt_->deltaTij()) *
            noise_model_between_bias_))); // imu between bias factor
    // add pose factor
    gtsam::Pose3 curPose = lidarPose.compose(imu2Base_);
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(
        X(key_), curPose, degenerate ? correction_noise2_ : correction_noise_);
    graph_factors_.add(pose_factor); // lidar pose factor

    // insert predicted values
    gtsam::NavState prop_state =
        imu_integrator_opt_->predict(prev_state_, prev_bias_);
    graph_values_.insert(X(key_), prop_state.pose());
    graph_values_.insert(V(key_), prop_state.v());
    graph_values_.insert(B(key_), prev_bias_);

    try
    {
      optimizer_.update(graph_factors_, graph_values_);
      optimizer_.update();
    }
    catch (gtsam::IndeterminantLinearSystemException e)
    {
      LOG(WARNING) << e.what();
      key_ = 1000;
      done_first_opt_ = false;
      return;
    }
    // graphFactors.resize(0);
    graph_factors_ = gtsam::NonlinearFactorGraph();
    graph_values_.clear();

    // Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer_.calculateEstimate();
    prev_pose_imu_in_local_ = result.at<gtsam::Pose3>(X(key_));
    prev_vel_ = result.at<gtsam::Vector3>(V(key_));
    prev_state_ = gtsam::NavState(prev_pose_imu_in_local_, prev_vel_);
    prev_bias_ = result.at<gtsam::imuBias::ConstantBias>(B(key_));

    // Reset the optimization preintegration object.
    imu_integrator_opt_->resetIntegrationAndSetBias(prev_bias_);

    // pub bias
    if (debug_)
    {
      Eigen::Quaterniond lidar_pose_q(r_w, r_x, r_y, r_z);
      Eigen::Vector3d lidar_pose_t(p_x, p_y, p_z);
      lidar_poses_.push_back(
          {lidar_pose_t, lidar_pose_q});
      Eigen::Quaterniond imu_pose_q(
          prop_state.pose().rotation().toQuaternion().w(),
          prop_state.pose().rotation().toQuaternion().x(),
          prop_state.pose().rotation().toQuaternion().y(),
          prop_state.pose().rotation().toQuaternion().z());
      Eigen::Vector3d imu_pose_t(prop_state.pose().x(), prop_state.pose().y(),
                                 prop_state.pose().z());

      imu_poses_.push_back({imu_pose_t, imu_pose_q});

      auto vec = prev_bias_.vector();
      geometry_msgs::PoseStamped tmp_msg;
      tmp_msg.header.stamp = ros::Time::now();
      tmp_msg.pose.position.x = vec[0];
      tmp_msg.pose.position.y = vec[1];
      tmp_msg.pose.position.z = vec[2];

      tmp_msg.pose.orientation.x = vec[3];
      tmp_msg.pose.orientation.y = vec[4];
      tmp_msg.pose.orientation.z = vec[5];

      bias_pub_.publish(tmp_msg);
    }

    if (debug_)
    {
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.stamp = ros::Time().fromSec(current_correction_time);
      pose_stamped.header.frame_id = odometryFrame;
      auto base_in_local = prev_pose_imu_in_local_.compose(base2Imu_);
      pose_stamped.pose.position.x = base_in_local.x();
      pose_stamped.pose.position.y = base_in_local.y();
      pose_stamped.pose.position.z = base_in_local.z();
      tf::Quaternion q = tf::createQuaternionFromRPY(
          base_in_local.rotation().roll(), base_in_local.rotation().pitch(),
          base_in_local.rotation().yaw());
      pose_stamped.pose.orientation.x = q.x();
      pose_stamped.pose.orientation.y = q.y();
      pose_stamped.pose.orientation.z = q.z();
      pose_stamped.pose.orientation.w = q.w();
      //       pub_pose_.publish(pose_stamped);

      global_path_.poses.push_back(pose_stamped);
      global_path_.header.frame_id = "odom";
      global_path_.header.stamp = pose_stamped.header.stamp;
      pub_path_.publish(global_path_);
    }

    if (failureDetection(prev_vel_, prev_bias_))
    {
      resetParams();
      return;
    }
    // 2. after optimization, re-propagate imu odometry preintegration
    prev_state_odom_ = prev_state_;
    prev_bias_odom_ = prev_bias_;
    // first pop imu message older than current correction data
    double lastImuQT = -1;
    while (!imu_que_imu_.empty() &&
           ROS_TIME(&imu_que_imu_.front()) < current_correction_time - delta_t_)
    {
      lastImuQT = ROS_TIME(&imu_que_imu_.front());
      imu_que_imu_.pop_front();
    }
    // repropogate
    if (!imu_que_imu_.empty())
    {
      // reset bias use the newly optimized bias
      imu_integrator_imu_->resetIntegrationAndSetBias(prev_bias_odom_);
      // integrate imu message from the beginning of this optimization
      for (int i = 0; i < (int)imu_que_imu_.size(); ++i)
      {
        sensor_msgs::Imu *thisImu = &imu_que_imu_[i];
        double imuTime = ROS_TIME(thisImu);
        double dt = (lastImuQT < 0) ? (1.0 / 500.0) : (imuTime - lastImuQT);

        imu_integrator_imu_->integrateMeasurement(
            gtsam::Vector3(thisImu->linear_acceleration.x,
                           thisImu->linear_acceleration.y,
                           thisImu->linear_acceleration.z),
            gtsam::Vector3(thisImu->angular_velocity.x,
                           thisImu->angular_velocity.y,
                           thisImu->angular_velocity.z),
            dt);
        lastImuQT = imuTime;
      }
    }
    ++key_;
    done_first_opt_ = true;

  }

	bool failureDetection(const gtsam::Vector3 &velCur,
				const gtsam::imuBias::ConstantBias &biasCur)
	{
		Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
		if (vel.norm() > 10)
		{
			LOG(WARNING) << ("Large velocity, reset IMU-preintegration!");
			return true;
			//             prevBias_ =
			//             gtsam::imuBias::ConstantBias({0,0,0},prevBias_.gyroscope());
		}

		Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(),
				biasCur.accelerometer().z());
		Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(),
				biasCur.gyroscope().z());
		if (ba.norm() > 1.5 || bg.norm() > 1.0)
		{
			LOG(WARNING) << ("Large bias, reset IMU-preintegration!");
			//             prevBias_ =
			//             gtsam::imuBias::ConstantBias(prevBias_.accelerometer()*0.5,prevBias_.gyroscope()*0.8);
			return true;
		}

		return false;
	}

	void imuHandler(const sensor_msgs::Imu::ConstPtr &imu_raw)
	{
	std::lock_guard<std::mutex> lock(mtx_);
	//      sensor_msgs::Imu thisImu = imuConverter(*imu_raw);
	if (!initialised_)
	{
		initialised_ = true;
		imu_tracker_.reset(new ImuTracker(10.0, imu_raw->header.stamp.toSec()));
	}
	sensor_msgs::Imu thisImu = imuConverter(*imu_raw, imu_tracker_);
	imu_que_opt_.push_back(thisImu);
	imu_que_imu_.push_back(thisImu);
	if (done_first_opt_ == false)
		return;

	double imuTime = ROS_TIME(&thisImu);
	double dt = (last_imu_time_imu_ < 0) ? (1.0 / 500.0) : (imuTime - last_imu_time_imu_);
	last_imu_time_imu_ = imuTime;
	// integrate this single imu message
	imu_integrator_imu_->integrateMeasurement(
		gtsam::Vector3(thisImu.linear_acceleration.x,
			thisImu.linear_acceleration.y,
			thisImu.linear_acceleration.z),
		gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y,
			thisImu.angular_velocity.z),
		dt);

	// predict odometry
	gtsam::NavState currentState =
		imu_integrator_imu_->predict(prev_state_odom_, prev_bias_odom_);

	// publish odometry
	nav_msgs::Odometry odometry;
	odometry.header.stamp = thisImu.header.stamp;
	odometry.header.frame_id = odometryFrame;
	odometry.child_frame_id = "odom_imu";

	// transform imu pose to lidar
	gtsam::Pose3 imuPose =
		gtsam::Pose3(currentState.quaternion(), currentState.position());
	gtsam::Pose3 pose_in_base = imuPose.compose(base2Imu_);

	odometry.pose.pose.position.x = pose_in_base.translation().x();
	odometry.pose.pose.position.y = pose_in_base.translation().y();
	odometry.pose.pose.position.z = pose_in_base.translation().z();
	odometry.pose.pose.orientation.x =
		pose_in_base.rotation().toQuaternion().x();
	odometry.pose.pose.orientation.y =
		pose_in_base.rotation().toQuaternion().y();
	odometry.pose.pose.orientation.z =
		pose_in_base.rotation().toQuaternion().z();
	odometry.pose.pose.orientation.w =
		pose_in_base.rotation().toQuaternion().w();
	if (isnan(odometry.pose.pose.orientation.x) || isnan(odometry.pose.pose.orientation.y) || isnan(odometry.pose.pose.orientation.z) || isnan(odometry.pose.pose.orientation.w) || (odometry.pose.pose.orientation.x == 0 && odometry.pose.pose.orientation.y == 0 && odometry.pose.pose.orientation.z == 0 && odometry.pose.pose.orientation.w == 0))
	{
		LOG(WARNING) << "dt: " << dt;
		LOG(WARNING) << "imuPose: " << imuPose;
		LOG(WARNING) << "prev_state_odom_: " << prev_state_odom_;
		LOG(WARNING) << "prev_bias_odom_: " << prev_bias_odom_;
		return;
	}
	odometry.twist.twist.linear.x = currentState.velocity().x();
	odometry.twist.twist.linear.y = currentState.velocity().y();
	odometry.twist.twist.linear.z = currentState.velocity().z();
	odometry.twist.twist.angular.x =
		thisImu.angular_velocity.x + prev_bias_odom_.gyroscope().x();
	odometry.twist.twist.angular.y =
		thisImu.angular_velocity.y + prev_bias_odom_.gyroscope().y();
	odometry.twist.twist.angular.z =
		thisImu.angular_velocity.z + prev_bias_odom_.gyroscope().z();
	pub_imu_odometry_.publish(odometry);
	
	}
};

int main(int argc, char **argv)
{
	google::InitGoogleLogging(argv[0]);
	google::SetStderrLogging(google::GLOG_INFO);
	FLAGS_colorlogtostderr = true;
	ros::init(argc, argv, "roboat_loam");

	IMUPreintegration ImuP;
	TransformFusion TF;


	ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");

	ros::MultiThreadedSpinner spinner(4);
	spinner.spin();

	return 0;
}
