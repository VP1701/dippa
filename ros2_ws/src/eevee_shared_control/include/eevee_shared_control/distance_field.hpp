// distance_field.hpp

#ifndef DISTANCE_FIELD_HPP
#define DISTANCE_FIELD_HPP

#include "status.hpp"
#include <Eigen/Core>                          // Eigen::Vector2d
#include <nav_msgs/msg/occupancy_grid.hpp>     // nav_msgs::msg::OccupancyGrid
#include <opencv2/core.hpp>

class DistanceField {

    private:
        cv::Mat sdf_;
        double resolution_ = 0.0;
        double inv_resolution_ = 0.0;
        Eigen::Vector2d origin_{Eigen::Vector2d::Zero()};
        double origin_yaw_ = 0.0;
        int width_ = 0;
        int height_ = 0;
        bool built_ = false;
        double cos_yaw_ = 0.0;
        double sin_yaw_ = 0.0;


    public:
        bool is_built() const;
        Status in_bounds(const Eigen::Vector2d&) const;
        Status build(const nav_msgs::msg::OccupancyGrid&, double occ_threshold);
        Status distance(const Eigen::Vector2d&, double& out) const;
        Status gradient(const Eigen::Vector2d&, Eigen::Vector2d&& out) const;
        Status distance_and_gradient(const Eigen::Vector2d&, double& out_dist, Eigen::Vector2d& grad) const;
        Status world_to_index(const Eigen::Vector2d&, Eigen::Vector2i&) const;

};

#endif // distance_field.hpp