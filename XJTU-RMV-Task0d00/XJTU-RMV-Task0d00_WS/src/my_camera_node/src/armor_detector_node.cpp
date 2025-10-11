#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>

// 新增：用于处理参数事件
#include <rclcpp/parameter_event_handler.hpp>

using namespace std;
using namespace cv;

class ArmorDetectorNode : public rclcpp::Node
{
public:
  ArmorDetectorNode() : Node("armor_detector_node")
  {
    // 1. 创建订阅者，订阅原始图像话题
    subscription_ = image_transport::create_subscription(
      this, "image_raw", 
      std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    // 2. 创建发布者，用于发布带有识别结果的图像
    result_image_pub_ = image_transport::create_publisher(this, "armor_detector/result_image");
    
    // ======================= 新增代码 [开始] =======================
    
    // 3. 声明并获取 `debug` 参数，用于控制是否开启调试模式
    // 默认设置为 true，方便您立即看到调试图像
    debug_ = this->declare_parameter("debug", true);

    // 如果 `debug` 模式开启，则创建调试相关的发布者
    if (debug_) {
      createDebugPublishers();
    }
    
    // 4. 设置参数回调，以便在运行时动态开启/关闭 debug 模式
    // 例如，您可以使用命令行: ros2 param set /armor_detector_node debug false
    param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    param_callback_handle_ = param_subscriber_->add_parameter_callback("debug",
      [this](const rclcpp::Parameter & p) {
        // 更新 debug 标志
        this->debug_ = p.as_bool();
        // 根据新的标志值，创建或销毁调试发布者
        if (this->debug_) {
          this->createDebugPublishers();
        } else {
          this->destroyDebugPublishers();
        }
      });
      
    // ======================= 新增代码 [结束] =======================
    
    RCLCPP_INFO(this->get_logger(), "Armor Detector Node has been started.");
  }

private:
  // 核心回调函数，每当接收到新图像时执行
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    try {
      // 将ROS图像消息转换为OpenCV图像格式
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      Mat frame = cv_ptr->image;
      
      // 调用装甲板检测函数
      Mat result_frame;
      // 修改：将消息头传入，以便调试信息的时间戳保持一致
      detectLightBar(frame, result_frame, msg->header); 

      // 将处理后的OpenCV图像转换回ROS消息并发布
      sensor_msgs::msg::Image::SharedPtr result_msg = 
        cv_bridge::CvImage(msg->header, "bgr8", result_frame).toImageMsg();
      result_image_pub_.publish(result_msg);
      
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  // 修改：函数签名增加一个 header 参数，用于发布调试消息
  void detectLightBar(const Mat& src, Mat &dst, const std_msgs::msg::Header & header) {
      dst = src.clone();
      
      // 1. 将原图从 BGR 转换到 HSV 色彩空间
      Mat hsv;
      cvtColor(src, hsv, COLOR_BGR2HSV);

      // 2. 定义 蓝色 装甲板的HSV阈值范围
      cv::Scalar lower_blue(100, 80, 80);
      cv::Scalar upper_blue(130, 255, 255);

      // 3. 创建一个二值化的掩码（Mask）
      Mat binary;
      cv::inRange(hsv, lower_blue, upper_blue, binary);

      // 4. 使用形态学操作清理掩码中的噪点
      Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
      morphologyEx(binary, binary, MORPH_OPEN, kernel);

      // ======================= 新增代码 [开始] =======================
      // 5. 如果开启了调试模式，则发布二值化掩码图
      if (debug_) {
        // 将单通道的二值图(binary)封装成ROS图像消息并发布
        // 注意编码格式为 "mono8"
        auto mask_msg = cv_bridge::CvImage(header, "mono8", binary).toImageMsg();
        binary_mask_pub_.publish(std::move(mask_msg));
      }
      // ======================= 新增代码 [结束] =======================

      vector<vector<Point>> contours;
      findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

      // ... 后续的灯条筛选和装甲板匹配代码保持不变 ...
      vector<RotatedRect> lightBars;
      for (const auto& cont : contours) {
        double area = contourArea(cont);
        if (area < 15 || area > 500) continue;
        if (cont.size() < 5) continue;
        RotatedRect r = minAreaRect(cont);
        float longSide = max(r.size.width, r.size.height);
        float shortSide = min(r.size.width, r.size.height);
        if (shortSide < 1e-5) continue;
        float ratio = longSide / shortSide;
        if (ratio < 1.5 || ratio > 10.0) continue;
        lightBars.push_back(r);
      }
      
      for (size_t i = 0; i < lightBars.size(); i++) {
          for (size_t j = i + 1; j < lightBars.size(); j++) {
              float angle_i = normalizedAngle(lightBars[i]);
              float angle_j = normalizedAngle(lightBars[j]);
              if (fabs(angle_i - angle_j) > 10.0) continue;
              float height_i = max(lightBars[i].size.width, lightBars[i].size.height);
              float height_j = max(lightBars[j].size.width, lightBars[j].size.height);
              if (fabs(height_i - height_j) / max(height_i, height_j) > 0.2) continue;
              Point2f center_diff = lightBars[i].center - lightBars[j].center;
              float distance = sqrt(center_diff.ddot(center_diff));
              float avg_height = (height_i + height_j) * 0.5;
              if(distance / avg_height < 1.0 || distance / avg_height > 5.0) continue;
              Point2f pts[4];
              RotatedRect armorRect = RotatedRect((lightBars[i].center + lightBars[j].center) * 0.5f,
                                                 Size2f(distance, avg_height), 
                                                 (angle_i + angle_j) * 0.5f);
              armorRect.points(pts);
              for (int k = 0; k < 4; k++) {
                  line(dst, pts[k], pts[(k + 1) % 4], Scalar(0, 255, 0), 2);
              }
          }
      }
  }

  // 和之前的代码一样
  static float normalizedAngle(const RotatedRect& r) {
      float w = r.size.width, h = r.size.height, ang = r.angle;
      if (w < h) ang += 90.0f;
      if (ang >= 180.0f) ang -= 180.0f;
      return ang;
  }

  // ======================= 新增代码 [开始] =======================
  void createDebugPublishers()
  {
    // 如果发布者还未创建，则创建它
    if (!binary_mask_pub_) {
      binary_mask_pub_ = image_transport::create_publisher(this, "armor_detector/binary_mask");
      RCLCPP_INFO(this->get_logger(), "Debug mode enabled. Publishing binary mask topic.");
    }
  }

  void destroyDebugPublishers()
  {
    // 如果发布者已存在，则销毁它
    if (binary_mask_pub_) {
      binary_mask_pub_.shutdown();
      // 在新版 image_transport 中，shutdown() 后最好将指针置空
      binary_mask_pub_ = {}; 
      RCLCPP_INFO(this->get_logger(), "Debug mode disabled. Stopped publishing binary mask topic.");
    }
  }
  // ======================= 新增代码 [结束] =======================

  // 成员变量
  image_transport::Subscriber subscription_;
  image_transport::Publisher result_image_pub_;

  // ======================= 新增代码 [开始] =======================
  bool debug_;
  image_transport::Publisher binary_mask_pub_;
  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> param_callback_handle_;
  // ======================= 新增代码 [结束] =======================
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmorDetectorNode>());
  rclcpp::shutdown();
  return 0;
}