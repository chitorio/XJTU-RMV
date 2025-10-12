#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/parameter_event_handler.hpp>

using namespace std;
using namespace cv;

class ArmorDetectorNode : public rclcpp::Node
{
public:
  ArmorDetectorNode() : Node("armor_detector_node")
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

    // 发布结果和掩码
    result_pub_ = image_transport::create_publisher(this, "armor_detector/result", qos.get_rmw_qos_profile());
    binary_pub_ = image_transport::create_publisher(this, "armor_detector/binary_mask", qos.get_rmw_qos_profile());
    lights_pub_ = image_transport::create_publisher(this, "armor_detector/lights", qos.get_rmw_qos_profile());
    
    // 参数
    this->declare_parameter("binary_thres", 180);
    this->declare_parameter("detect_color", 1); // 0: RED, 1: BLUE
    this->declare_parameter("debug", true);
    this->declare_parameter("min_contour_area", 50);
    this->declare_parameter("max_contour_area", 3000);
    this->declare_parameter("min_lightbar_ratio", 2.0);
    this->declare_parameter("max_lightbar_ratio", 8.0);
    this->declare_parameter("max_angle_diff", 15.0);
    this->declare_parameter("max_height_diff", 0.3);
    
    // 加载参数
    binary_thres_ = this->get_parameter("binary_thres").as_int();
    detect_color_ = this->get_parameter("detect_color").as_int();
    debug_ = this->get_parameter("debug").as_bool();
    min_contour_area_ = this->get_parameter("min_contour_area").as_int();
    max_contour_area_ = this->get_parameter("max_contour_area").as_int();
    min_lightbar_ratio_ = this->get_parameter("min_lightbar_ratio").as_double();
    max_lightbar_ratio_ = this->get_parameter("max_lightbar_ratio").as_double();
    max_angle_diff_ = this->get_parameter("max_angle_diff").as_double();
    max_height_diff_ = this->get_parameter("max_height_diff").as_double();

    RCLCPP_INFO(this->get_logger(), "Armor Detector started - Color: %s", detect_color_ == 0 ? "RED" : "BLUE");
  }

private:
  struct Light {
    cv::RotatedRect rect;
    cv::Point2f top, bottom;
    double length;
    double width;
    cv::Point2f center;
    float tilt_angle;
    int color;
    
    Light(cv::RotatedRect r) : rect(r) {
      width = min(r.size.width, r.size.height);
      length = max(r.size.width, r.size.height);
      center = r.center;
      
      // 计算端点
      float angle_rad = r.angle * CV_PI / 180.0f;
      float dx = length * 0.5f * cos(angle_rad);
      float dy = length * 0.5f * sin(angle_rad);
      top = cv::Point2f(center.x - dx, center.y - dy);
      bottom = cv::Point2f(center.x + dx, center.y + dy);
      
      tilt_angle = r.angle;
      if (tilt_angle > 90.0f) tilt_angle = 180.0f - tilt_angle;
    }
  };

  void detectCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
      const cv::Mat& frame = cv_ptr->image;
      
      // 预处理
      cv::Mat binary_img = preprocessImage(frame);
      
      // 发布掩码图像
      if (binary_pub_.getNumSubscribers() > 0) {
        auto mask_msg = cv_bridge::CvImage(msg->header, "mono8", binary_img).toImageMsg();
        binary_pub_.publish(mask_msg);
      }
      
      // 查找灯条
      std::vector<Light> lights = findLights(binary_img);
      
      // 发布灯条可视化结果
      publishLightVisualization(frame, lights, msg->header);
      
      // 匹配装甲板
      std::vector<std::pair<Light, Light>> armors = matchLights(lights);
      
      // 发布最终结果
      publishResult(frame, armors, msg->header);
      
      // 性能统计
      logPerformance(start_time, armors.size(), lights.size());
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
    }
  }

  cv::Mat preprocessImage(const cv::Mat& rgb_img)
  {
    cv::Mat gray, binary;
    
    // 转换为灰度图
    cv::cvtColor(rgb_img, gray, cv::COLOR_BGR2GRAY);
    
    // 二值化
    cv::threshold(gray, binary, binary_thres_, 255, cv::THRESH_BINARY);
    
    // 形态学操作去噪
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    
    return binary;
  }

  std::vector<Light> findLights(const cv::Mat& binary_img)
  {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<Light> lights;
    
    for (const auto& contour : contours) {
      double area = cv::contourArea(contour);
      if (area < min_contour_area_ || area > max_contour_area_) continue;
      
      cv::RotatedRect rect = cv::minAreaRect(contour);
      float width = min(rect.size.width, rect.size.height);
      float length = max(rect.size.width, rect.size.height);
      
      if (width < 1e-5) continue;
      
      // 长宽比筛选
      float ratio = length / width;
      if (ratio < min_lightbar_ratio_ || ratio > max_lightbar_ratio_) continue;
      
      // 角度筛选
      float angle = rect.angle;
      if (angle > 90.0f) angle = 180.0f - angle;
      if (angle > 30.0f) continue;
      
      lights.emplace_back(rect);
    }
    
    return lights;
  }

  std::vector<std::pair<Light, Light>> matchLights(const std::vector<Light>& lights)
  {
    std::vector<std::pair<Light, Light>> armors;
    
    for (size_t i = 0; i < lights.size(); i++) {
      for (size_t j = i + 1; j < lights.size(); j++) {
        const Light& l1 = lights[i];
        const Light& l2 = lights[j];
        
        if (isValidArmorPair(l1, l2)) {
          armors.emplace_back(l1, l2);
        }
      }
    }
    
    return armors;
  }

  bool isValidArmorPair(const Light& l1, const Light& l2)
  {
    // 高度一致性
    float height_ratio = min(l1.length, l2.length) / max(l1.length, l2.length);
    if (height_ratio < 0.6f) return false;
    
    // 距离合理性
    float distance = cv::norm(l1.center - l2.center);
    float avg_length = (l1.length + l2.length) * 0.5f;
    float distance_ratio = distance / avg_length;
    if (distance_ratio < 1.0f || distance_ratio > 4.0f) return false;
    
    // 角度差
    float angle_diff = abs(l1.tilt_angle - l2.tilt_angle);
    if (angle_diff > max_angle_diff_) return false;
    
    // 水平对齐
    float y_diff = abs(l1.center.y - l2.center.y);
    if (y_diff > avg_length * max_height_diff_) return false;
    
    return true;
  }

  void publishLightVisualization(const cv::Mat& frame, const std::vector<Light>& lights, const std_msgs::msg::Header& header)
  {
    if (lights_pub_.getNumSubscribers() == 0) return;
    
    cv::Mat lights_frame = frame.clone();
    
    for (const auto& light : lights) {
      // 绘制灯条轮廓
      cv::Point2f vertices[4];
      light.rect.points(vertices);
      
      // 用黄色绘制灯条边界框
      for (int i = 0; i < 4; i++) {
        cv::line(lights_frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
      }
      
      // 绘制灯条中心线（顶部到底部）
      cv::line(lights_frame, light.top, light.bottom, cv::Scalar(255, 0, 0), 3);
      
      // 绘制中心点
      cv::circle(lights_frame, light.center, 4, cv::Scalar(0, 0, 255), -1);
      
      // 显示灯条信息
      std::string info = "L:" + std::to_string((int)light.length) + " A:" + std::to_string((int)light.tilt_angle);
      cv::putText(lights_frame, info, cv::Point(light.center.x + 10, light.center.y), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
    
    // 添加统计信息
    std::string count_info = "Lights: " + std::to_string(lights.size());
    cv::putText(lights_frame, count_info, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    auto lights_msg = cv_bridge::CvImage(header, "bgr8", lights_frame).toImageMsg();
    lights_pub_.publish(lights_msg);
  }

  void publishResult(const cv::Mat& frame, const std::vector<std::pair<Light, Light>>& armors, 
                    const std_msgs::msg::Header& header)
  {
    if (result_pub_.getNumSubscribers() == 0) return;
    
    cv::Mat result_frame = frame.clone();
    
    // 首先绘制所有灯条（用浅色）
    std::vector<Light> all_lights;
    for (const auto& armor : armors) {
      all_lights.push_back(armor.first);
      all_lights.push_back(armor.second);
    }
    
    for (const auto& light : all_lights) {
      cv::line(result_frame, light.top, light.bottom, cv::Scalar(0, 200, 200), 2); // 青色灯条
      cv::circle(result_frame, light.center, 3, cv::Scalar(0, 200, 200), -1);
    }
    
    // 绘制装甲板
    for (const auto& armor : armors) {
      const Light& l1 = armor.first;
      const Light& l2 = armor.second;
      
      // 绘制装甲板矩形（绿色）
      std::vector<cv::Point2f> armor_points = {
        l1.top, l2.top, l2.bottom, l1.bottom
      };
      
      for (int i = 0; i < 4; i++) {
        cv::line(result_frame, armor_points[i], armor_points[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
      }
      
      // 绘制装甲板中心点（红色）
      cv::Point2f center = (l1.center + l2.center) * 0.5f;
      cv::circle(result_frame, center, 6, cv::Scalar(0, 0, 255), -1);
      
      // 显示配对信息
      std::string pair_info = "Armor";
      cv::putText(result_frame, pair_info, center + cv::Point2f(10, 0), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    }
    
    // 添加调试信息
    if (debug_) {
      std::string info = "Armors: " + std::to_string(armors.size());
      cv::putText(result_frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
      
      // 显示参数信息
      std::string param_info = "Thres: " + std::to_string(binary_thres_);
      cv::putText(result_frame, param_info, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
    
    auto result_msg = cv_bridge::CvImage(header, "bgr8", result_frame).toImageMsg();
    result_pub_.publish(result_msg);
  }

  void logPerformance(const std::chrono::high_resolution_clock::time_point& start_time, int armor_count, int light_count)
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    frame_count_++;
    total_time_ += duration.count();
    
    if (frame_count_ >= 30) {
      double fps = 1000000.0 / (total_time_ / frame_count_);
      RCLCPP_INFO(this->get_logger(), "FPS: %.1f, Lights: %d, Armors: %d", fps, light_count, armor_count);
      frame_count_ = 0;
      total_time_ = 0;
    }
  }

  // 成员变量
  image_transport::Subscriber subscription_;
  image_transport::Publisher result_pub_;
  image_transport::Publisher binary_pub_;
  image_transport::Publisher lights_pub_;
  
  // 参数
  int binary_thres_;
  int detect_color_;
  bool debug_;
  int min_contour_area_;
  int max_contour_area_;
  double min_lightbar_ratio_;
  double max_lightbar_ratio_;
  double max_angle_diff_;
  double max_height_diff_;
  
  // 性能统计
  int frame_count_ = 0;
  long long total_time_ = 0;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmorDetectorNode>());
  rclcpp::shutdown();
  return 0;
}