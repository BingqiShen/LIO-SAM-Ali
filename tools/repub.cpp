#include <iostream>
#include<fstream>

#include<ros/ros.h>
#include<pcl_conversions/pcl_conversions.h>
#include<sensor_msgs/PointCloud2.h>
#include<sensor_msgs/NavSatFix.h>
#include<gps_common/GPSFix.h>
#include<sensor_msgs/Imu.h>


#include <tf2_msgs/TFMessage.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/LinearMath/Quaternion.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Core>

#include <math.h>

using namespace std;

void rtkCallback(const gps_common::GPSFix::ConstPtr& msg, ros::Publisher *gps_pub)
{
	sensor_msgs::NavSatFix gps_fix;
	gps_fix.header = msg->header;
	gps_fix.status.status = msg->status.status;
	gps_fix.latitude = msg->latitude;
	gps_fix.longitude = msg->longitude;
	gps_fix.altitude = msg->altitude;
	gps_fix.position_covariance = msg->position_covariance;
	gps_fix.position_covariance_type = msg->position_covariance_type;
	gps_fix.position_covariance_type = 1;

	gps_pub->publish(gps_fix);
}

void imuCallback(const sensor_msgs::Imu::ConstPtr& msg, ros::Publisher *imu_pub)
{
	sensor_msgs::Imu imu_raw;
	imu_raw = *msg;
	imu_raw.header.frame_id = "imu_link";


	imu_pub->publish(imu_raw);
}


 
int main(int argc,char **argv)
{
	ros::init(argc, argv, "readTF");
	ros::NodeHandle nh;
	ros::Publisher gps_pub = nh.advertise<sensor_msgs::NavSatFix>("/gps/fix", 10);
	ros::Publisher imu_pub = nh.advertise<sensor_msgs::Imu>("/imu_raw", 400);

	ros::Subscriber sub_rtk = nh.subscribe<gps_common::GPSFix>
								("/rtk", 10, boost::bind(&rtkCallback, _1, &gps_pub));
	ros::Subscriber sub_imu = nh.subscribe<sensor_msgs::Imu>
								("/imu/data", 400, boost::bind(&imuCallback, _1, &imu_pub));
    
    ros::Rate loop_rate(400);
    while(ros::ok())
	{
        ros::spinOnce();
        loop_rate.sleep();
    }

	return 0;
}
