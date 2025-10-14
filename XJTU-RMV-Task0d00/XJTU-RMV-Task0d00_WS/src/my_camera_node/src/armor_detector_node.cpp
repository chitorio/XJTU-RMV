#include <memory>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/parameter_event_handler.hpp>

using namespace std;
using namespace cv;

class ArmorDetectorNode : public rclcpp::Node
{
public:
  ArmorDetectorNode() : Node("armor_detector_node"), camera_info_received_(false)
  {
    // 初始化时间戳
    last_frame_stamp_.sec = 0;
    last_frame_stamp_.nanosec = 0;
    
    // 高性能QoS
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    
    // 订阅图像和相机信息
    subscription_ = image_transport::create_subscription(
      this, "image_raw", 
      [this](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        this->detectCallback(msg);
      },
      "raw", qos.get_rmw_qos_profile());

    // 订阅相机信息
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "camera_info", qos,
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        this->cameraInfoCallback(msg);
      });

    // 发布结果话题（删除其他调试话题）
    result_pub_ = image_transport::create_publisher(this, "armor_detector/result", qos.get_rmw_qos_profile());
    
    // 发布装甲板3D坐标
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("armor_detector/armor_poses", 10);
    
    // 声明所有可调参数
    this->declare_parameter("binary_thres", 200);
    this->declare_parameter("detect_color", 1); // 0: RED, 1: BLUE
    this->declare_parameter("debug", true);
    this->declare_parameter("min_contour_area", 20);
    this->declare_parameter("max_contour_area", 12000);
    this->declare_parameter("min_lightbar_ratio", 1.5);
    this->declare_parameter("max_lightbar_ratio", 30.0);
    this->declare_parameter("max_angle_diff", 30.0);
    this->declare_parameter("max_height_diff_ratio", 0.7);
    this->declare_parameter("max_tilt_angle", 60.0);
    this->declare_parameter("armor_height_extension", 0);
    this->declare_parameter("armor_width", 0.135);
    this->declare_parameter("armor_height", 0.056);

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
    armor_width_ = this->get_parameter("armor_width").as_double();
    armor_height_ = this->get_parameter("armor_height").as_double();

    // 初始化装甲板3D模型点
    initArmor3DPoints();

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

  struct Armor {
    Light left_light;
    Light right_light;
    cv::RotatedRect rect;
    cv::Point2f center;
    std::vector<cv::Point2f> corners;
    cv::Point3f camera_coordinates;
    bool valid_pose;
    
    Armor(const Light& l1, const Light& l2, const cv::RotatedRect& r) 
      : left_light(l1), right_light(l2), rect(r), valid_pose(false) {
      center = (l1.center + l2.center) * 0.5f;
    }
  };

  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (camera_info_received_) {
      return;
    }

    // 提取相机内参矩阵
    camera_matrix_ = (cv::Mat_<double>(3, 3) <<
      msg->k[0], msg->k[1], msg->k[2],
      msg->k[3], msg->k[4], msg->k[5],
      msg->k[6], msg->k[7], msg->k[8]);

    // 提取畸变系数
    dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
    for (size_t i = 0; i < msg->d.size() && i < 5; ++i) {
      dist_coeffs_.at<double>(i) = msg->d[i];
    }

    camera_info_received_ = true;
    
    RCLCPP_INFO(this->get_logger(), "Camera info received - fx: %.1f, fy: %.1f", 
                camera_matrix_.at<double>(0,0), camera_matrix_.at<double>(1,1));
  }

  // MODIFIED FUNCTION
  void initArmor3DPoints()
  {
    // 装甲板3D模型点
    // 坐标系: X轴向前, Y轴向左, Z轴向上
    // 单位: 米
    double half_y = armor_width_ / 2.0;
    double half_z = armor_height_ / 2.0;
    
    armor_points_3d_.clear();
    // 顺序: 左下, 左上, 右上, 右下
    armor_points_3d_.emplace_back(cv::Point3f(0, half_y, -half_z));
    armor_points_3d_.emplace_back(cv::Point3f(0, half_y, half_z));
    armor_points_3d_.emplace_back(cv::Point3f(0, -half_y, half_z));
    armor_points_3d_.emplace_back(cv::Point3f(0, -half_y, -half_z));
  }

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
      std::vector<Armor> armors = matchLights(lights);
      
      // PnP解算
      if (camera_info_received_ && !armors.empty()) {
        solvePnPForArmors(armors);
        publishArmorCoordinates(armors, msg->header);
      }
      
      // 只发布最终结果（删除其他调试发布）
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
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    
    return binary;
  }

  std::vector<Light> findLights(const cv::Mat& binary_img)
  {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Light> lights;
    
    for (const auto& contour : contours) {
      double area = cv::contourArea(contour);
      if (area < min_contour_area_) continue;
      
      cv::RotatedRect rect = cv::minAreaRect(contour);
      cv::Rect bbox = cv::boundingRect(contour);
      
      Light light(rect, bbox, area);
      
      if (std::abs(light.tilt_angle) > max_tilt_angle_) continue;
      if (light.width < 1e-5) continue;
      
      float ratio = light.length / light.width;
      if (ratio < min_lightbar_ratio_ || ratio > max_lightbar_ratio_) continue;
      
      lights.emplace_back(light);
    }
    return lights;
  }

  std::vector<Armor> matchLights(const std::vector<Light>& lights)
  {
    std::vector<Armor> armors;
    for (size_t i = 0; i < lights.size(); i++) {
      for (size_t j = i + 1; j < lights.size(); j++) {
        const Light& l1 = lights[i];
        const Light& l2 = lights[j];
        if (isValidArmorPair(l1, l2)) {
          // 确定左右灯条
          const Light& left_light = (l1.center.x < l2.center.x) ? l1 : l2;
          const Light& right_light = (l1.center.x < l2.center.x) ? l2 : l1;
          
          // 创建装甲板旋转矩形
          cv::RotatedRect armor_rect = createArmorRect(left_light, right_light);
          Armor armor(left_light, right_light, armor_rect);
          
          // 获取装甲板四个角点
          armor.corners = getArmorCorners(armor_rect);
          armors.emplace_back(armor);
        }
      }
    }
    return armors;
  }

  cv::RotatedRect createArmorRect(const Light& left_light, const Light& right_light)
  {
    cv::Point2f center = (left_light.center + right_light.center) * 0.5f;
    float armor_width = cv::norm(left_light.center - right_light.center);
    float avg_height = (left_light.length + right_light.length) * 0.5f;
    float armor_height = avg_height * (1.0f + armor_height_extension_);
    float armor_angle = (left_light.tilt_angle + right_light.tilt_angle) * 0.5f;
    
    return cv::RotatedRect(center, cv::Size2f(armor_width, armor_height), armor_angle);
  }

  std::vector<cv::Point2f> getArmorCorners(const cv::RotatedRect& rect)
  {
    cv::Point2f vertices[4];
    rect.points(vertices);
    return std::vector<cv::Point2f>(vertices, vertices + 4);
  }

  bool isValidArmorPair(const Light& l1, const Light& l2)
  {
    float avg_length = (l1.length + l2.length) * 0.5f;
    
    float distance = cv::norm(l1.center - l2.center);
    float distance_ratio = distance / avg_length;
    if (distance_ratio < 0.8f || distance_ratio > 6.0f) return false;
    
    float angle_diff = std::abs(l1.tilt_angle - l2.tilt_angle);
    if (angle_diff > max_angle_diff_) return false;
    
    float y_diff = std::abs(l1.center.y - l2.center.y);
    if (y_diff > avg_length * max_height_diff_ratio_) return false;
    
    float dot_product = std::abs(std::cos(l1.tilt_angle * CV_PI / 180.0f) * 
                                std::cos(l2.tilt_angle * CV_PI / 180.0f) +
                                std::sin(l1.tilt_angle * CV_PI / 180.0f) * 
                                std::sin(l2.tilt_angle * CV_PI / 180.0f));
    if (dot_product < 0.5f) return false;
    
    return true;
  }

  // MODIFIED FUNCTION
  // [最终诊断版本] 检查所有输入的 solvePnPForArmors 函数
  void solvePnPForArmors(std::vector<Armor>& armors)
  {
    for (auto& armor : armors) {
      // 2. 准备2D点
      std::vector<cv::Point2f> image_points;
      image_points.emplace_back(armor.left_light.bottom);
      image_points.emplace_back(armor.left_light.top);
      image_points.emplace_back(armor.right_light.top);
      image_points.emplace_back(armor.right_light.bottom);

      // 3. 在调用前打印日志，确认输入是正常的
      // RCLCPP_INFO(this->get_logger(), "Attempting PnP with %zu image points.", image_points.size());
      // 你可以取消下面这行注释，看到具体的像素坐标
      // for(const auto& p : image_points) { RCLCPP_INFO(this->get_logger(), "  - (%.1f, %.1f)", p.x, p.y); }

      cv::Mat rvec, tvec;
      bool success = false;
      
      try {
        success = cv::solvePnP(armor_points_3d_, image_points, camera_matrix_, 
                              dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
        
        if (success) {
          RCLCPP_INFO(this->get_logger(), "PnP Succeeded!");
          double x = tvec.at<double>(0);
          double y = tvec.at<double>(1); 
          double z = tvec.at<double>(2);
          armor.camera_coordinates = cv::Point3f(x, y, z);
          armor.valid_pose = true;
        }

      } catch (const cv::Exception& e) {
        RCLCPP_FATAL(this->get_logger(), "PnP CRASHED: cv::solvePnP threw an exception: %s", e.what());
        armor.valid_pose = false;
      }
    }
  }

  // 重投影误差计算函数
  double calculateReprojectionError(const std::vector<cv::Point3f>& object_points,
                                   const std::vector<cv::Point2f>& image_points,
                                   const cv::Mat& rvec, const cv::Mat& tvec)
  {
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, projected_points);
    
    double total_error = 0.0;
    for (size_t i = 0; i < image_points.size(); i++) {
      total_error += cv::norm(image_points[i] - projected_points[i]);
    }
    return total_error / image_points.size();
  }

  void publishArmorCoordinates(const std::vector<Armor>& armors, const std_msgs::msg::Header& header)
  {
    if (pose_pub_->get_subscription_count() == 0) return;
    
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header = header;
    
    for (const auto& armor : armors) {
      if (!armor.valid_pose) continue;
      
      geometry_msgs::msg::Pose pose;
      pose.position.x = armor.camera_coordinates.x;
      pose.position.y = armor.camera_coordinates.y;
      pose.position.z = armor.camera_coordinates.z;
      // Nota: La orientación no se calcula aquí. Se deja como identidad.
      pose.orientation.w = 1.0; 
      pose_array.poses.push_back(pose);
    }
    
    if (!pose_array.poses.empty()) {
      pose_pub_->publish(pose_array);
    }
  }

  void publishResult(const cv::Mat& frame, const std::vector<Armor>& armors, const std_msgs::msg::Header& header)
  {
    if (result_pub_.getNumSubscribers() == 0) return;
    cv::Mat result_frame = frame.clone();
    for (const auto& armor : armors) {
      // 绘制装甲板区域
      cv::Point2f vertices[4];
      armor.rect.points(vertices);
      for (int i = 0; i < 4; i++) {
        cv::line(result_frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
      }
      
      // 绘制3D坐标信息
      if (armor.valid_pose) {
        std::string coord_text = cv::format("(%.2f,%.2f,%.2f)m", 
                                          armor.camera_coordinates.x,
                                          armor.camera_coordinates.y,
                                          armor.camera_coordinates.z);
        cv::putText(result_frame, coord_text, armor.center + cv::Point2f(10, -10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
      }
      
      cv::circle(result_frame, armor.center, 5, cv::Scalar(0, 0, 255), -1);
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
      // 简化性能输出
      RCLCPP_INFO(this->get_logger(), "Frame processed - Lights: %d, Armors: %d", light_count, armor_count);
      frame_count_ = 0;
      total_time_ = 0;
    }
  }

  // 成员变量（删除调试发布器）
  image_transport::Subscriber subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  image_transport::Publisher result_pub_;  // 只保留结果发布器
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_pub_;
  
  int binary_thres_, detect_color_, min_contour_area_, max_contour_area_;
  bool debug_;
  double min_lightbar_ratio_, max_lightbar_ratio_, max_angle_diff_, max_height_diff_ratio_, max_tilt_angle_;
  double armor_height_extension_;
  double armor_width_, armor_height_;
  
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  std::vector<cv::Point3f> armor_points_3d_;
  bool camera_info_received_;
  
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