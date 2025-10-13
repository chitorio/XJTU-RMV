#include <memory>
#include <opencv2/imgproc.hpp>
#include <rclcpp/qos.hpp>
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
    // 初始化时间戳
    last_frame_stamp_.sec = 0;
    last_frame_stamp_.nanosec = 0;
    
    // 高性能QoS
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    
    // 订阅图像
    subscription_ = image_transport::create_subscription(
      this, "image_raw", 
      [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        this->detectCallback(msg);
      },
      "raw", qos.get_rmw_qos_profile());

    // 发布结果和调试话题
    result_pub_ = image_transport::create_publisher(this, "armor_detector/result", qos.get_rmw_qos_profile());
    binary_pub_ = image_transport::create_publisher(this, "armor_detector/binary_mask", qos.get_rmw_qos_profile());
    lights_pub_ = image_transport::create_publisher(this, "armor_detector/lights", qos.get_rmw_qos_profile());
    bbox_pub_ = image_transport::create_publisher(this, "armor_detector/bounding_boxes", qos.get_rmw_qos_profile());
    
    // 声明所有可调参数
    this->declare_parameter("binary_thres", 180);
    this->declare_parameter("detect_color", 1); // 0: RED, 1: BLUE
    this->declare_parameter("debug", true);
    this->declare_parameter("min_contour_area", 20);
    this->declare_parameter("max_contour_area", 6000);
    this->declare_parameter("min_lightbar_ratio", 1.5);
    this->declare_parameter("max_lightbar_ratio", 15.0);
    this->declare_parameter("max_angle_diff", 30.0);  // 放宽角度差到30度
    this->declare_parameter("max_height_diff_ratio", 0.7);  // 放宽高度差比例
    this->declare_parameter("max_tilt_angle", 60.0);  // 放宽最大倾斜角度到60度
    this->declare_parameter("armor_height_extension", 0.3);  // 新增：装甲板高度延伸比例

    // 加载参数
    binary_thres_ = this->get_parameter("binary_thres").as_int();
    detect_color_ = this->get_parameter("detect_color").as_int();
    debug_ = this->get_parameter("debug").as_bool();
    min_contour_area_ = this->get_parameter("min_contour_area").as_int();
    max_contour_area_ = this->get_parameter("max_contour_area").as_int();
    min_lightbar_ratio_ = this->get_parameter("min_lightbar_ratio").as_double();
    max_lightbar_ratio_ = this->get_parameter("max_lightbar_ratio").as_double();
    max_angle_diff_ = this->get_parameter("max_angle_diff").as_double();
    max_height_diff_ratio_ = this->get_parameter("max_height_diff_ratio").as_double();
    max_tilt_angle_ = this->get_parameter("max_tilt_angle").as_double();
    armor_height_extension_ = this->get_parameter("armor_height_extension").as_double();

    RCLCPP_INFO(this->get_logger(), "Armor Detector started - Color: %s", detect_color_ == 0 ? "RED" : "BLUE");
  }

private:
  struct Light {
    cv::RotatedRect rect;
    cv::Rect bbox;
    cv::Point2f top, bottom;
    double length;
    double width;
    cv::Point2f center;
    float tilt_angle;  // 与水平方向的夹角 [-90, 90]
    int color;
    double area;
    
    Light(cv::RotatedRect r, cv::Rect b, double a) : rect(r), bbox(b), area(a) {
      center = r.center;
      
      // 修复角度计算：使用OpenCV的标准角度定义
      // OpenCV的minAreaRect角度范围是[-90, 0)，0度是水平的
      width = r.size.width;
      length = r.size.height;
      tilt_angle = r.angle;
      
      // 如果宽度大于高度，说明矩形是横向的，需要调整
      if (r.size.width > r.size.height) {
        width = r.size.height;
        length = r.size.width;
        tilt_angle = r.angle + 90.0f;
      }
      
      // 将角度归一化到 [-90, 90] 范围
      while (tilt_angle > 90.0f) tilt_angle -= 180.0f;
      while (tilt_angle < -90.0f) tilt_angle += 180.0f;
      
      // 计算端点 - 更简单直接的方法
      float angle_rad = tilt_angle * CV_PI / 180.0f;
      float dx = length * 0.5f * std::cos(angle_rad);
      float dy = length * 0.5f * std::sin(angle_rad);
      
      top = cv::Point2f(center.x - dx, center.y - dy);
      bottom = cv::Point2f(center.x + dx, center.y + dy);
      
      // 确保top在bottom上方
      if (top.y > bottom.y) {
        std::swap(top, bottom);
      }
    }
  };

  void detectCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    // 时间戳去重
    if (msg->header.stamp.sec == last_frame_stamp_.sec && 
        msg->header.stamp.nanosec == last_frame_stamp_.nanosec) {
      return;
    }
    last_frame_stamp_ = msg->header.stamp;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
      const cv::Mat& frame = cv_ptr->image;
      
      cv::Mat binary_img = preprocessImage(frame);
      std::vector<Light> lights = findLights(binary_img);
      std::vector<std::pair<Light, Light>> armors = matchLights(lights);
      
      // 调试信息发布
      if (debug_) {
        publishBoundingBoxes(frame, lights, msg->header);
        publishLightVisualization(frame, lights, msg->header);
        if (binary_pub_.getNumSubscribers() > 0) {
            auto mask_msg = cv_bridge::CvImage(msg->header, "mono8", binary_img).toImageMsg();
            binary_pub_.publish(std::move(mask_msg));
        }
      }
      
      publishResult(frame, armors, msg->header);
      logPerformance(start_time, armors.size(), lights.size());
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
    }
  }

  cv::Mat preprocessImage(const cv::Mat& rgb_img)
  {
    cv::Mat gray, binary;
    cv::cvtColor(rgb_img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, binary_thres_, 255, cv::THRESH_BINARY);
    
    // 形态学操作
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
      cv::Rect bbox = cv::boundingRect(contour);
      
      Light light(rect, bbox, area);
      
      // 放宽倾斜角度限制
      if (std::abs(light.tilt_angle) > max_tilt_angle_) continue;
      if (light.width < 1e-5) continue;
      
      float ratio = light.length / light.width;
      if (ratio < min_lightbar_ratio_ || ratio > max_lightbar_ratio_) continue;
      
      lights.emplace_back(light);
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
    float avg_length = (l1.length + l2.length) * 0.5f;

    // 放宽高度一致性
    float height_ratio = std::min(l1.length, l2.length) / std::max(l1.length, l2.length);
    if (height_ratio < 0.4f) return false;  // 从0.6放宽到0.4
    
    // 放宽距离合理性
    float distance = cv::norm(l1.center - l2.center);
    float distance_ratio = distance / avg_length;
    if (distance_ratio < 0.8f || distance_ratio > 6.0f) return false;  // 放宽距离范围
    
    // 放宽角度差
    float angle_diff = std::abs(l1.tilt_angle - l2.tilt_angle);
    if (angle_diff > max_angle_diff_) return false;
    
    // 放宽水平对齐
    float y_diff = std::abs(l1.center.y - l2.center.y);
    if (y_diff > avg_length * max_height_diff_ratio_) return false;
    
    // 放宽方向一致性
    float dot_product = std::abs(std::cos(l1.tilt_angle * CV_PI / 180.0f) * 
                                std::cos(l2.tilt_angle * CV_PI / 180.0f) +
                                std::sin(l1.tilt_angle * CV_PI / 180.0f) * 
                                std::sin(l2.tilt_angle * CV_PI / 180.0f));
    if (dot_product < 0.5f) return false; // 从0.7放宽到0.5
    
    return true;
  }

  void publishBoundingBoxes(const cv::Mat& frame, const std::vector<Light>& lights, const std_msgs::msg::Header& header)
  {
    if (bbox_pub_.getNumSubscribers() == 0) return;
    cv::Mat bbox_frame = frame.clone();
    for (const auto& light : lights) {
      cv::rectangle(bbox_frame, light.bbox, cv::Scalar(255, 0, 0), 2);
      cv::Point2f vertices[4];
      light.rect.points(vertices);
      for (int i = 0; i < 4; i++) {
        cv::line(bbox_frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
      }
      // 显示角度信息
      std::string angle_info = std::to_string((int)light.tilt_angle);
      cv::putText(bbox_frame, angle_info, light.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    auto bbox_msg = cv_bridge::CvImage(header, "bgr8", bbox_frame).toImageMsg();
    bbox_pub_.publish(std::move(bbox_msg));
  }

  void publishLightVisualization(const cv::Mat& frame, const std::vector<Light>& lights, const std_msgs::msg::Header& header)
  {
    if (lights_pub_.getNumSubscribers() == 0) return;
    cv::Mat lights_frame = frame.clone();
    for (const auto& light : lights) {
      cv::Point2f vertices[4];
      light.rect.points(vertices);
      for (int i = 0; i < 4; i++) {
        cv::line(lights_frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
      }
      cv::line(lights_frame, light.top, light.bottom, cv::Scalar(255, 0, 0), 3);
      // 绘制角度方向
      cv::arrowedLine(lights_frame, light.center, light.bottom, cv::Scalar(0, 0, 255), 2);
    }
    auto lights_msg = cv_bridge::CvImage(header, "bgr8", lights_frame).toImageMsg();
    lights_pub_.publish(std::move(lights_msg));
  }

  void publishResult(const cv::Mat& frame, const std::vector<std::pair<Light, Light>>& armors, const std_msgs::msg::Header& header)
  {
    if (result_pub_.getNumSubscribers() == 0) return;
    cv::Mat result_frame = frame.clone();
    for (const auto& armor : armors) {
      const Light& l1 = armor.first;
      const Light& l2 = armor.second;
      
      // 新的装甲板区域绘制方法
      drawArmorRegion(result_frame, l1, l2);
    }
    auto result_msg = cv_bridge::CvImage(header, "bgr8", result_frame).toImageMsg();
    result_pub_.publish(std::move(result_msg));
  }

  void drawArmorRegion(cv::Mat& frame, const Light& l1, const Light& l2)
  {
    // 计算两个灯条之间的中心点
    cv::Point2f center = (l1.center + l2.center) * 0.5f;
    
    // 计算装甲板宽度（灯条之间的距离）
    float armor_width = cv::norm(l1.center - l2.center);
    
    // 计算装甲板高度（取两个灯条高度的平均值，并上下延伸一定比例）
    float avg_height = (l1.length + l2.length) * 0.5f;
    float armor_height = avg_height * (1.0f + armor_height_extension_);
    
    // 计算装甲板的角度（使用两个灯条角度的平均值）
    float armor_angle = (l1.tilt_angle + l2.tilt_angle) * 0.5f;
    
    // 创建装甲板的旋转矩形
    cv::RotatedRect armor_rect(center, cv::Size2f(armor_width, armor_height), armor_angle);
    
    // 绘制装甲板区域
    cv::Point2f vertices[4];
    armor_rect.points(vertices);
    for (int i = 0; i < 4; i++) {
      cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
    }
    
    // 可选：在装甲板中心绘制标记
    cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), -1);
  }

  void logPerformance(const std::chrono::high_resolution_clock::time_point& start_time, int armor_count, int light_count)
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    frame_count_++;
    total_time_ += duration.count();
    if (frame_count_ >= 30) {
      RCLCPP_INFO(this->get_logger(), "Lights Found: %d, Armors Found: %d", light_count, armor_count);
      frame_count_ = 0;
      total_time_ = 0;
    }
  }

  // 成员变量
  image_transport::Subscriber subscription_;
  image_transport::Publisher result_pub_, binary_pub_, lights_pub_, bbox_pub_;
  
  int binary_thres_, detect_color_, min_contour_area_, max_contour_area_;
  bool debug_;
  double min_lightbar_ratio_, max_lightbar_ratio_, max_angle_diff_, max_height_diff_ratio_, max_tilt_angle_;
  double armor_height_extension_;  // 装甲板高度延伸比例
  
  int frame_count_ = 0;
  long long total_time_ = 0;

  builtin_interfaces::msg::Time last_frame_stamp_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<ArmorDetectorNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("rclcpp"), "Node creation failed: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}