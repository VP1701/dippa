// distance_field.cpp

#include "eevee_shared_control/distance_field.hpp"

#include <opencv2/imgproc.hpp>   // header only declares core
#include <cmath>
#include <geometry_msgs/msg/quaternion.hpp>

Status DistanceField::build(const nav_msgs::msg::OccupancyGrid& msg, double occ_threshold) {
    built_ = false;
    const int w = static_cast<int>(msg.info.width);
    const int h = static_cast<int>(msg.info.height);

    const double res = static_cast<double>(msg.info.resolution);

    if (w <= 0 || h <= 0 || res <= 0.0) {
        return Status::Error;
    }

    if(msg.data.size() != static_cast<size_t>(msg.info.width) * static_cast<size_t>(msg.info.height)) {
        return Status::DimensionMismatch;
    }

    width_ = w;
    height_ = h;
    resolution_ = res;
    inv_resolution_ = 1.0 / res;
    origin_ << msg.info.origin.position.x, msg.info.origin.position.y;

    const geometry_msgs::msg::Quaternion& q = msg.info.origin.orientation;
    origin_yaw_ = std::atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z);
    cos_yaw_ = std::cos(origin_yaw_);
    sin_yaw_ = std::sin(origin_yaw_);

    // create occupancy mask

    const cv::Mat grid(h, w, CV_8SC1, const_cast<int8_t * >(msg.data.data()));

    cons double thr = std::clamp(occ_threshold, 0.0, 1.0) * 100.0;

    cv::Mat occupied = (grid >= thr);


    // distance transforms

    cv::Mat d_out;
    cv::Mat d_in;
    
}