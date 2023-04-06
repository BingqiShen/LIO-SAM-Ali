#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <iostream>
#include <fstream>

#include<pcl_conversions/pcl_conversions.h>
#include<sensor_msgs/PointCloud2.h>
#include<sensor_msgs/NavSatFix.h>
#include<gps_common/GPSFix.h>

#include <tf2_msgs/TFMessage.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/LinearMath/Quaternion.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace message_filters;

#define DEGREES_TO_RADIANS M_PI / 180
#define alt 416.0
#define WGS84_A 6378137.0
#define f 1.0 / 298.257223563
#define WGS84_B WGS84_A * (1.0 - f)
#define WGS84_E sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_A
#define WGS84_E2 sqrt(WGS84_A * WGS84_A - WGS84_B * WGS84_B) / WGS84_B
#define NAV_E2 (2.0 - f) * f // also e^2

std::vector<std::vector<double>> ecef_coordinate;
std::vector<std::vector<double>> odom_coordinate;


void tfCallback(const tf2_msgs::TFMessage::ConstPtr& msg, ros::Publisher *odom_pub)
{
	static tf::TransformListener listener;
	static tf::StampedTransform transform;

	try
	{
		listener.waitForTransform("odom", "base_link", msg->transforms[0].header.stamp, ros::Duration(0.1));
		listener.lookupTransform("odom", "base_link", msg->transforms[0].header.stamp, transform);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
	
	double x, y, z, timestamp;
	timestamp = msg->transforms[0].header.stamp.toSec();
	x = transform.getOrigin().x();
	y = transform.getOrigin().y();
	z = transform.getOrigin().z();

	Eigen::Quaterniond q;
	q.x() = transform.getRotation().getX();
	q.y() = transform.getRotation().getY();
	q.z() = transform.getRotation().getZ();
	q.w() = transform.getRotation().getW();

	nav_msgs::Odometry odometry;
	odometry.header.stamp = msg->transforms[0].header.stamp;
	odometry.header.frame_id = "odom";
	odometry.child_frame_id = "base_link";

	odometry.pose.pose.position.x = x;
	odometry.pose.pose.position.y = y;
	odometry.pose.pose.position.z = z;
	odometry.pose.pose.orientation.x = q.x();
	odometry.pose.pose.orientation.y = q.y();
	odometry.pose.pose.orientation.z = q.z();
	odometry.pose.pose.orientation.w = q.w();

	odom_pub->publish(odometry);

}

void latlon2ecef(double lat, double lon, double *ecef_x, double *ecef_y, double *ecef_z)
{
	double clat, slat, clon, slon, N;
    clat = cos(lat * DEGREES_TO_RADIANS);
    slat = sin(lat * DEGREES_TO_RADIANS);
    clon = cos(lon * DEGREES_TO_RADIANS);
    slon = sin(lon * DEGREES_TO_RADIANS);

    N = WGS84_A / sqrt(1.0 - WGS84_E * WGS84_E * slat * slat);

    *ecef_x = (N + alt) * clat * clon;
    *ecef_y = (N + alt) * clat * slon;
    *ecef_z = (N * (1.0 - WGS84_E * WGS84_E) + alt) * slat;
}

void ecef2latlon(double ecef_x, double ecef_y, double ecef_z, double *lat, double *lon)
{
	double longtitude_rad, latitude_rad;
	longtitude_rad = atan2(ecef_y, ecef_x);
    *lon = longtitude_rad / M_PI * 180;

    // latitude_rad = asin(sqrt((WGS84_A * WGS84_A * sin(longtitude_rad) * sin(longtitude_rad) - ecef_y * ecef_y)
    //                             / (WGS84_A * WGS84_A * sin(longtitude_rad) * sin(longtitude_rad) + (f * f - 2 * f) * ecef_y * ecef_y) ));
	double r = sqrt(ecef_x*ecef_x + ecef_y*ecef_y);
    

	double templat = atan2(ecef_z, r);
	double tempalt = sqrt(r * r + ecef_z * ecef_z) - WGS84_A;
	double rhoerror = 1000.0;
	double zerror = 1000.0;
        
	int iter = 0; // number of iterations

	//      %  Newton's method iteration on templat and tempalt makes
	//      %   the residuals on rho and z progressively smaller.  Loop
	//      %   is implemented as a 'while' instead of a 'do' to simplify
	//      %   porting to MATLAB

	while ((abs(rhoerror) > 1e-6) | (abs(zerror) > 1e-6)) 
	{
		double slat = sin(templat);
		double clat = cos(templat);
		double q = 1.0 - NAV_E2 * slat * slat;
		double r_n = WGS84_A / sqrt(q);
		double drdl = r_n * NAV_E2 * slat * clat / q; // d(r_n)/d(latitutde)

		rhoerror = (r_n + tempalt) * clat - r;
		zerror = (r_n * (1.0 - NAV_E2) + tempalt) * slat - ecef_z;

		//          %             --                               -- --      --
		//          %             |  drhoerror/dlat  drhoerror/dalt | |  a  b  |
		//                        % Find Jacobian           |                       |=|        |
		//          %             |   dzerror/dlat    dzerror/dalt  | |  c  d  |
		//          %             --                               -- --      --

		double aa = drdl * clat - (r_n + tempalt) * slat;
		double bb = clat;
		double cc = (1.0 - NAV_E2)*(drdl * slat + r_n * clat);
		double dd = slat;

		//Apply correction = inv(Jacobian)*errorvector

		double invdet = 1.0 / (aa * dd - bb * cc);
		templat = templat - invdet * (+dd * rhoerror - bb * zerror);
		tempalt = tempalt - invdet * (-cc * rhoerror + aa * zerror);

		iter++;

		if (iter>20)
		{
			std::cout << "xyz2lla could not converge" << std::endl;
			return;
		}
	}

	*lat = templat / M_PI * 180;
}

void callback(const sensor_msgs::NavSatFixConstPtr& gps_msg, const nav_msgs::OdometryConstPtr& odom_msg, ros::Publisher *base2gps_pub)
{
	// cout << "gps_msg->header.stamp.now(): " << gps_msg->header.stamp.now() << endl;

	if(ecef_coordinate.size() == odom_coordinate.size())
	{
		// 1. ecef
		double ecef_x, ecef_y, ecef_z;
		latlon2ecef(gps_msg->latitude, gps_msg->longitude, &ecef_x, &ecef_y, &ecef_z);
		
		// 2. odom
		double odom_x, odom_y, odom_z;
		odom_x = odom_msg->pose.pose.position.x;
		odom_y = odom_msg->pose.pose.position.y;
		odom_z = odom_msg->pose.pose.position.z;

		cout << "ecef_coordinate.size(): " << ecef_coordinate.size() << endl;

		const int estimate_num = 50;
		if(ecef_coordinate.size() < estimate_num)
		{
			std::vector<double> ecef_xyz;
			ecef_xyz.push_back(ecef_x);
			ecef_xyz.push_back(ecef_y);
			ecef_xyz.push_back(ecef_z);
			ecef_coordinate.push_back(ecef_xyz);

			std::vector<double> odom_xyz;
			odom_xyz.push_back(odom_x);
			odom_xyz.push_back(odom_y);
			odom_xyz.push_back(odom_z);
			odom_coordinate.push_back(odom_xyz);

			return;
		}

		Eigen::Vector3d cur_odom(odom_x, odom_y, odom_z);
		Eigen::Matrix<double, 3, Eigen::Dynamic> cloud_tgt(3, estimate_num);
  		Eigen::Matrix<double, 3, Eigen::Dynamic> cloud_src(3, estimate_num);

		for(int i = 0; i < ecef_coordinate.size(); i++)
		{
			cloud_tgt(0, i) = odom_coordinate[i][0];
			cloud_tgt(1, i) = odom_coordinate[i][1];
			cloud_tgt(2, i) = odom_coordinate[i][2];

			cloud_src(0, i) = ecef_coordinate[i][0];
			cloud_src(1, i) = ecef_coordinate[i][1];
			cloud_src(2, i) = ecef_coordinate[i][2];
		}
		
		

		Eigen::Matrix4d st = Eigen::umeyama(cloud_src, cloud_tgt, false);
		Eigen::Matrix4d ts = Eigen::umeyama(cloud_tgt, cloud_src, false);

		std::cout << "------------------------------------------------------------" << std::endl;

		std::cout << "-------------------------ecef to odom-------------------------" << std::endl;

		std::cout << "------------------------------------------------------------" << std::endl;

		std::cout << st << std::endl;

		std::cout << "------------------------------------------------------------" << std::endl;

		std::cout << "-------------------------odom to ecef-------------------------" << std::endl;

		std::cout << "------------------------------------------------------------" << std::endl;

		std::cout << ts << std::endl;

		Eigen::Vector3d t_odom2ecef(ts(0, 3), ts(1, 3), ts(2, 3));
		Eigen::Vector3d t_ecef = ts.block<3, 3>(0, 0) * cur_odom + t_odom2ecef;

		// cout << t_ecef.transpose() << endl;

		double cur_lat, cur_lon;
		ecef2latlon(t_ecef(0), t_ecef(1), t_ecef(2), &cur_lat, &cur_lon);
		cout << std::fixed << "cur_lat: " << cur_lat << " cur_lon: " << cur_lon << endl;

		nav_msgs::Odometry gps_odometry;
		gps_odometry.header.stamp = gps_msg->header.stamp;
		gps_odometry.header.frame_id = "gnss_link";
		gps_odometry.child_frame_id = "base_link";

		gps_odometry.pose.pose.position.x = cur_lat;
		gps_odometry.pose.pose.position.y = cur_lon;
		gps_odometry.pose.pose.position.z = 0.0;
		gps_odometry.pose.pose.orientation.x = 0.0;
		gps_odometry.pose.pose.orientation.y = 0.0;
		gps_odometry.pose.pose.orientation.z = 0.0;
		gps_odometry.pose.pose.orientation.w = 1.0;

		base2gps_pub->publish(gps_odometry);
		
	}
	else
	{
		cout << "wrong!!!" << endl;
	}

	
}

 
int main(int argc,char **argv)
{
	ros::init(argc, argv, "gps_odom_align");
	ros::NodeHandle nh;

	ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/base2odom", 10);
	ros::Subscriber sub_tf = nh.subscribe<tf2_msgs::TFMessage>
								("/tf", 10, boost::bind(&tfCallback, _1, &odom_pub));
	
	ros::Publisher base2gps_pub = nh.advertise<nav_msgs::Odometry>("/base2gps", 10);

	message_filters::Subscriber<sensor_msgs::NavSatFix> gps_sub(nh, "/gps/fix", 1);
	message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/base2odom", 1);

	typedef sync_policies::ApproximateTime<sensor_msgs::NavSatFix, nav_msgs::Odometry> GPS_Odom_Sync_Policy;
	message_filters::Synchronizer<GPS_Odom_Sync_Policy> sync(GPS_Odom_Sync_Policy(10), gps_sub, odom_sub);
	sync.registerCallback(boost::bind(&callback, _1, _2, &base2gps_pub));

    
    // ros::Rate loop_rate(1);
    // while(ros::ok()){
    //     ros::spinOnce();
    //     loop_rate.sleep();
    // }

	ros::spin();

	return 0;
}
