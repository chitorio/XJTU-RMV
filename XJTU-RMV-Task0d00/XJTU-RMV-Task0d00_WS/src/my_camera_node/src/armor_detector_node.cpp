#include <memory>
#include <opencv2/imgproc.hpp>
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
    this->declare_parameter("max_contour_area", 3000);
    this->declare_parameter("min_lightbar_ratio", 1.0);
    this->declare_parameter("max_lightbar_ratio", 20.0);
    this->declare_parameter("max_angle_diff", 10.0);
    this->declare_parameter("max_height_diff_ratio", 0.8); // 使用比例代替绝对值
    
    // 加载参数 (可以放在回调中实时更新，但为简化先在构造时加载)
    binary_thres_ = this->get_parameter("binary_thres").as_int();
    detect_color_ = this->get_parameter("detect_color").as_int();
    debug_ = this->get_parameter("debug").as_bool();
    min_contour_area_ = this->get_parameter("min_contour_area").as_int();
    max_contour_area_ = this->get_parameter("max_contour_area").as_int();
    min_lightbar_ratio_ = this->get_parameter("min_lightbar_ratio").as_double();
    max_lightbar_ratio_ = this->get_parameter("max_lightbar_ratio").as_double();
    max_angle_diff_ = this->get_parameter("max_angle_diff").as_double();
    max_height_diff_ratio_ = this->get_parameter("max_height_diff_ratio").as_double();

    RCLCPP_INFO(this->get_logger(), "Armor Detector started - Color: %s", detect_color_ == 0 ? "RED" : "BLUE");
  }

private:
  // ============================ 关键数据结构 ============================
  struct Light {
    cv::RotatedRect rect;       // 原始最小外接矩形
    cv::Rect bbox;              // 垂直外接矩形
    cv::Point2f top, bottom;    // 灯条上下端点
    double length;              // 灯条长度 (归一化后)
    double width;               // 灯条宽度 (归一化后)
    cv::Point2f center;         // 中心点
    float tilt_angle;           // 与垂直方向的锐角 [0, 90)
    int color;                  // 颜色 (暂未实现)
    double area;                // 轮廓面积
    
    // 构造函数：在此完成所有角度和尺寸的归一化，确保数据一致性
    Light(cv::RotatedRect r, cv::Rect b, double a) : rect(r), bbox(b), area(a) {
      center = r.center;
      
      // 1. 保证 height 是长边, width 是短边
      if (r.size.width > r.size.height) {
        length = r.size.width;
        width = r.size.height;
        tilt_angle = r.angle + 90.0f;
      } else {
        length = r.size.height;
        width = r.size.width;
        tilt_angle = r.angle;
      }
      
      // 2. 将角度归一化为与垂直Y轴的锐角 [0, 45] 度
      // OpenCV角度范围是[-90, 0)，-90为垂直
      // 我们想要的是灯条与垂直方向的夹角
      tilt_angle = std::abs(90 + tilt_angle); // 转换为与垂直方向的角度
      if (tilt_angle > 90) {
        tilt_angle = 180 - tilt_angle;
      }

      // 3. 鲁棒地计算上下端点
      cv::Point2f vertices[4];
      r.points(vertices);
      float max_dist = 0;
      int p1_idx = -1, p2_idx = -1;
      for (int i=0; i<4; ++i) {
          for (int j=i+1; j<4; ++j) {
              float d = cv::norm(vertices[i] - vertices[j]);
              if (d > max_dist) {
                  max_dist = d;
                  p1_idx = i;
                  p2_idx = j;
              }
          }
      }
      // 保证 top 在 bottom 的上方 (y值更小)
      if (vertices[p1_idx].y < vertices[p2_idx].y) {
          top = vertices[p1_idx];
          bottom = vertices[p2_idx];
      } else {
          top = vertices[p2_idx];
          bottom = vertices[p1_idx];
      }
    }
  };

  // ============================ 核心处理流程 ============================
  void detectCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
      const cv::Mat& frame = cv_ptr->image;
      
      cv::Mat binary_img = preprocessImage(frame);
      std::vector<Light> lights = findLights(binary_img);
      std::vector<std::pair<Light, Light>> armors = matchLights(lights);
      
      // 只有在开启debug模式时才发布调试图像
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

  // ============================ 算法实现 ============================
  cv::Mat preprocessImage(const cv::Mat& rgb_img)
  {
    cv::Mat gray, binary;
    cv::cvtColor(rgb_img, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, binary_thres_, 255, cv::THRESH_BINARY);
    
    // 开闭操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
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
      
      if (light.tilt_angle > 30.0f) continue;
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

    float height_ratio = min(l1.length, l2.length) / max(l1.length, l2.length);
    if (height_ratio < 0.7f) return false;
    
    float distance = cv::norm(l1.center - l2.center);
    float distance_ratio = distance / avg_length;
    if (distance_ratio < 1.0f || distance_ratio > 4.5f) return false;
    
    float angle_diff = abs(l1.tilt_angle - l2.tilt_angle);
    if (angle_diff > max_angle_diff_) return false;
    
    float y_diff = abs(l1.center.y - l2.center.y);
    if (y_diff > avg_length * max_height_diff_ratio_) return false;
    
    return true;
  }

  // ============================ 可视化与日志 ============================
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
      cv::line(lights_frame, light.top, light.bottom, cv::Scalar(255, 0, 0), 2);
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
      std::vector<cv::Point2f> armor_points = { l1.top, l2.top, l2.bottom, l1.bottom };
      for (int i = 0; i < 4; i++) {
        cv::line(result_frame, armor_points[i], armor_points[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
      }
    }
    auto result_msg = cv_bridge::CvImage(header, "bgr8", result_frame).toImageMsg();
    result_pub_.publish(std::move(result_msg));
  }

  void logPerformance(const std::chrono::high_resolution_clock::time_point& start_time, int armor_count, int light_count)
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    frame_count_++;
    total_time_ += duration.count();
    if (frame_count_ >= 30) {
      double fps = 1000000.0 / (total_time_ / frame_count_);
      RCLCPP_INFO(this->get_logger(), "FPS: %.1f, Lights Found: %d, Armors Found: %d", fps, light_count, armor_count);
      frame_count_ = 0;
      total_time_ = 0;
    }
  }

  // ============================ 成员变量 ============================
  image_transport::Subscriber subscription_;
  image_transport::Publisher result_pub_, binary_pub_, lights_pub_, bbox_pub_;
  
  int binary_thres_, detect_color_, min_contour_area_, max_contour_area_;
  bool debug_;
  double min_lightbar_ratio_, max_lightbar_ratio_, max_angle_diff_, max_height_diff_ratio_;
  
  int frame_count_ = 0;
  long long total_time_ = 0;
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