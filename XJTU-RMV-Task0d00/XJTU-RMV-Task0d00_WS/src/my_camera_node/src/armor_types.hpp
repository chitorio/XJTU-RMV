#ifndef ARMOR_TYPES_HPP_
#define ARMOR_TYPES_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 定义灯条结构体
struct Light {
  cv::RotatedRect rect;
  cv::Rect bbox;
  cv::Point2f top, bottom;
  double length;
  double width;
  cv::Point2f center;
  float tilt_angle;
  int color;
  double area;
  
  Light(cv::RotatedRect r, cv::Rect b, double a) : rect(r), bbox(b), area(a) {
    center = r.center;
    
    width = r.size.width;
    length = r.size.height;
    tilt_angle = r.angle;
    
    if (r.size.width > r.size.height) {
      width = r.size.height;
      length = r.size.width;
      tilt_angle = r.angle + 90.0f;
    }
    
    while (tilt_angle > 90.0f) tilt_angle -= 180.0f;
    while (tilt_angle < -90.0f) tilt_angle += 180.0f;
    
    float angle_rad = tilt_angle * CV_PI / 180.0f;
    float dx = length * 0.5f * std::cos(angle_rad);
    float dy = length * 0.5f * std::sin(angle_rad);
    
    top = cv::Point2f(center.x - dx, center.y - dy);
    bottom = cv::Point2f(center.x + dx, center.y + dy);
    
    if (top.y > bottom.y) {
      std::swap(top, bottom);
    }
  }
};

// 定义装甲板结构体
struct Armor {
  Light left_light;
  Light right_light;
  cv::RotatedRect rect;
  cv::Point2f center;
  std::vector<cv::Point2f> corners;
  cv::Point3f camera_coordinates;
  bool valid_pose;

  // 新增数字识别信息
  std::string detected_number;
  double number_confidence;
  bool number_valid;
  
  Armor(const Light& l1, const Light& l2, const cv::RotatedRect& r) 
    : left_light(l1), right_light(l2), rect(r), valid_pose(false),
      number_valid(false), number_confidence(0.0) {
    center = (l1.center + l2.center) * 0.5f;
  }
};

#endif // ARMOR_TYPES_HPP_