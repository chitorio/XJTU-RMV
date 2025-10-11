#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class MultiSourceCameraNode : public rclcpp::Node
{
public:
  MultiSourceCameraNode() : Node("multi_source_camera_node")
  {
    // 参数声明
    this->declare_parameter<std::string>("source_mode", "video_file");
    this->declare_parameter<std::string>("video_path", "");
    this->declare_parameter<int>("camera_id", 0);
    this->declare_parameter<double>("frame_rate", 30.0);
    this->declare_parameter<bool>("loop_video", true);
    this->declare_parameter<bool>("resize", true);
    this->declare_parameter<int>("resize_width", 640);
    this->declare_parameter<int>("resize_height", 480);

    // 获取参数
    std::string source_mode = this->get_parameter("source_mode").as_string();
    std::string video_path = this->get_parameter("video_path").as_string();
    int camera_id = this->get_parameter("camera_id").as_int();
    double frame_rate = this->get_parameter("frame_rate").as_double();
    bool loop_video = this->get_parameter("loop_video").as_bool();
    bool resize = this->get_parameter("resize").as_bool();
    int resize_width = this->get_parameter("resize_width").as_int();
    int resize_height = this->get_parameter("resize_height").as_int();

    // 初始化视频源
    if (source_mode == "video_file") {
      if (video_path.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Video path is empty!");
        rclcpp::shutdown();
        return;
      }
      cap_.open(video_path);
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open video: %s", video_path.c_str());
        rclcpp::shutdown();
        return;
      }
    } else if (source_mode == "usb_camera") {
      cap_.open(camera_id);
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open USB camera: %d", camera_id);
        rclcpp::shutdown();
        return;
      }
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unknown source mode: %s", source_mode.c_str());
      rclcpp::shutdown();
      return;
    }

    // 创建发布器
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    publisher_ = image_transport::create_publisher(this, "image_raw", qos.get_rmw_qos_profile());

    // 定时器
    int interval_ms = static_cast<int>(1000.0 / frame_rate);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(interval_ms),
      std::bind(&MultiSourceCameraNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "MultiSourceCameraNode started - Mode: %s", source_mode.c_str());
  }

private:
  void timerCallback()
  {
    cv::Mat frame;
    if (!cap_.read(frame)) {
      if (this->get_parameter("loop_video").as_bool() && 
          this->get_parameter("source_mode").as_string() == "video_file") {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!cap_.read(frame)) {
          RCLCPP_ERROR(this->get_logger(), "Failed to restart video");
          return;
        }
      } else {
        RCLCPP_INFO(this->get_logger(), "End of video stream");
        return;
      }
    }

    // 调整大小
    if (this->get_parameter("resize").as_bool()) {
      cv::resize(frame, frame, cv::Size(
        this->get_parameter("resize_width").as_int(),
        this->get_parameter("resize_height").as_int()
      ));
    }

    // 发布图像
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = this->now();
    msg->header.frame_id = "camera_optical_frame";
    publisher_.publish(msg);
  }

  cv::VideoCapture cap_;
  image_transport::Publisher publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MultiSourceCameraNode>());
  rclcpp::shutdown();
  return 0;
}