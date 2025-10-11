#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class FastArmorDetector : public rclcpp::Node
{
public:
  FastArmorDetector() : Node("fast_armor_detector")
  {
    // 高性能QoS
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    
    // 订阅图像
    subscription_ = image_transport::create_subscription(
      this, "image_raw", 
      [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        this->detectCallback(msg);
      },
      "raw", qos.get_rmw_qos_profile());

    // 发布结果
    result_pub_ = image_transport::create_publisher(this, "armor_detector/result", qos.get_rmw_qos_profile());
    
    // 参数
    this->declare_parameter("binary_thres", 150);
    this->declare_parameter("detect_color", 1); // 0: RED, 1: BLUE
    this->declare_parameter("debug", false);
    
    binary_thres_ = this->get_parameter("binary_thres").as_int();
    detect_color_ = this->get_parameter("detect_color").as_int();
    debug_ = this->get_parameter("debug").as_bool();

    RCLCPP_INFO(this->get_logger(), "Fast Armor Detector started");
  }

private:
  struct Light {
    cv::Rect rect;
    cv::Point2f top, bottom;
    double length;
    double width;
    cv::Point2f center;
    float tilt_angle;
    int color; // 0: RED, 1: BLUE
    
    Light(cv::Rect r, cv::Point2f t, cv::Point2f b, double l, double w, float angle)
      : rect(r), top(t), bottom(b), length(l), width(w), tilt_angle(angle)
    {
      center = (top + bottom) * 0.5f;
    }
  };

  struct Armor {
    Light left_light, right_light;
    int type; // 0: SMALL, 1: LARGE
    
    Armor(const Light& l1, const Light& l2) : left_light(l1), right_light(l2) {}
  };

  void detectCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
      const cv::Mat& frame = cv_ptr->image;
      
      // 预处理
      cv::Mat binary_img = preprocessImage(frame);
      
      // 查找灯条
      std::vector<Light> lights = findLights(frame, binary_img);
      
      // 匹配装甲板
      std::vector<Armor> armors = matchLights(lights);
      
      // 发布结果
      publishResult(frame, armors, msg->header);
      
      // 性能统计
      logPerformance(start_time);
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
    }
  }

  cv::Mat preprocessImage(const cv::Mat& rgb_img)
  {
    cv::Mat gray_img, binary_img;
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img, binary_img, binary_thres_, 255, cv::THRESH_BINARY);
    return binary_img;
  }

  std::vector<Light> findLights(const cv::Mat& rgb_img, const cv::Mat& binary_img)
  {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<Light> lights;
    
    for (const auto& contour : contours) {
      if (contour.size() < 5) continue;
      
      double area = cv::contourArea(contour);
      if (area < 50 || area > 5000) continue;
      
      cv::RotatedRect r_rect = cv::minAreaRect(contour);
      cv::Rect b_rect = cv::boundingRect(contour);
      
      // 计算灯条参数
      float width = std::min(r_rect.size.width, r_rect.size.height);
      float length = std::max(r_rect.size.width, r_rect.size.height);
      float ratio = length / width;
      
      // 筛选灯条
      if (ratio < 1.5f || ratio > 10.0f) continue;
      
      // 计算灯条端点
      cv::Point2f top, bottom;
      if (r_rect.size.width < r_rect.size.height) {
        top = cv::Point2f(r_rect.center.x, r_rect.center.y - length/2);
        bottom = cv::Point2f(r_rect.center.x, r_rect.center.y + length/2);
      } else {
        top = cv::Point2f(r_rect.center.x - length/2, r_rect.center.y);
        bottom = cv::Point2f(r_rect.center.x + length/2, r_rect.center.y);
      }
      
      float angle = r_rect.angle;
      if (angle > 90.0f) angle = 180.0f - angle;
      
      Light light(b_rect, top, bottom, length, width, angle);
      
      // 颜色识别
      if (isValidLight(light, rgb_img, contour)) {
        lights.push_back(light);
      }
    }
    
    return lights;
  }

  bool isValidLight(const Light& light, const cv::Mat& rgb_img, const std::vector<cv::Point>& contour)
  {
    // 角度筛选
    if (light.tilt_angle > 30.0f) return false;
    
    // 长宽比筛选
    float ratio = light.length / light.width;
    if (ratio < 2.0f || ratio > 8.0f) return false;
    
    // 颜色识别
    cv::Mat mask = cv::Mat::zeros(light.rect.size(), CV_8UC1);
    std::vector<cv::Point> mask_contour;
    for (const auto& p : contour) {
      mask_contour.emplace_back(p - cv::Point(light.rect.x, light.rect.y));
    }
    cv::fillPoly(mask, {mask_contour}, 255);
    
    int sum_r = 0, sum_b = 0, count = 0;
    auto roi = rgb_img(light.rect);
    
    for (int i = 0; i < roi.rows; i++) {
      for (int j = 0; j < roi.cols; j++) {
        if (mask.at<uchar>(i, j) > 0) {
          sum_b += roi.at<cv::Vec3b>(i, j)[0]; // Blue channel
          sum_r += roi.at<cv::Vec3b>(i, j)[2]; // Red channel
          count++;
        }
      }
    }
    
    if (count == 0) return false;
    
    // 根据检测颜色筛选
    if (detect_color_ == 0) { // RED
      return sum_r > sum_b * 1.2;
    } else { // BLUE
      return sum_b > sum_r * 1.2;
    }
  }

  std::vector<Armor> matchLights(const std::vector<Light>& lights)
  {
    std::vector<Armor> armors;
    
    for (size_t i = 0; i < lights.size(); i++) {
      for (size_t j = i + 1; j < lights.size(); j++) {
        const Light& l1 = lights[i];
        const Light& l2 = lights[j];
        
        if (isValidArmor(l1, l2)) {
          armors.emplace_back(Armor(l1, l2));
        }
      }
    }
    
    return armors;
  }

  bool isValidArmor(const Light& l1, const Light& l2)
  {
    // 高度差
    float height_ratio = std::min(l1.length, l2.length) / std::max(l1.length, l2.length);
    if (height_ratio < 0.7f) return false;
    
    // 距离
    float distance = cv::norm(l1.center - l2.center);
    float avg_length = (l1.length + l2.length) / 2.0f;
    float distance_ratio = distance / avg_length;
    
    if (distance_ratio < 1.0f || distance_ratio > 5.0f) return false;
    
    // 角度差
    float angle_diff = std::abs(l1.tilt_angle - l2.tilt_angle);
    if (angle_diff > 15.0f) return false;
    
    // 水平对齐
    float y_diff = std::abs(l1.center.y - l2.center.y);
    if (y_diff > avg_length * 0.5f) return false;
    
    return true;
  }

  void publishResult(const cv::Mat& frame, const std::vector<Armor>& armors, const std_msgs::msg::Header& header)
  {
    if (result_pub_.getNumSubscribers() == 0) return;
    
    cv::Mat result_frame = frame.clone();
    
    // 绘制装甲板
    for (const auto& armor : armors) {
      cv::line(result_frame, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
      cv::line(result_frame, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
      
      // 绘制中心
      cv::Point2f center = (armor.left_light.center + armor.right_light.center) * 0.5f;
      cv::circle(result_frame, center, 4, cv::Scalar(0, 0, 255), -1);
    }
    
    auto result_msg = cv_bridge::CvImage(header, "bgr8", result_frame).toImageMsg();
    result_pub_.publish(result_msg);
  }

  void logPerformance(const std::chrono::high_resolution_clock::time_point& start_time)
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    frame_count_++;
    total_time_ += duration.count();
    
    if (frame_count_ >= 30) {
      double fps = 1000000.0 / (total_time_ / frame_count_);
      RCLCPP_INFO(this->get_logger(), "Detection FPS: %.1f", fps);
      frame_count_ = 0;
      total_time_ = 0;
    }
  }

  image_transport::Subscriber subscription_;
  image_transport::Publisher result_pub_;
  
  int binary_thres_;
  int detect_color_;
  bool debug_;
  
  int frame_count_ = 0;
  long long total_time_ = 0;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FastArmorDetector>());
  rclcpp::shutdown();
  return 0;
}