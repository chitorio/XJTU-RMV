#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

class MultiSourceCameraNode : public rclcpp::Node
{
public:
  MultiSourceCameraNode() : Node("multi_source_camera_node"), stop_capture_(false)
  {
    // 参数声明
    this->declare_parameter<std::string>("source_mode", "video_file"); // "video_file" 或 "usb_camera"
    this->declare_parameter<std::string>("video_path", "");
    this->declare_parameter<int>("camera_id", 0);
    this->declare_parameter<double>("frame_rate", 30.0);
    this->declare_parameter<bool>("loop_video", true);
    this->declare_parameter<bool>("resize", true);
    this->declare_parameter<int>("resize_width", 640);
    this->declare_parameter<int>("resize_height", 480);

    // 获取参数
    source_mode_ = this->get_parameter("source_mode").as_string();
    video_path_ = this->get_parameter("video_path").as_string();
    camera_id_ = this->get_parameter("camera_id").as_int();
    frame_rate_ = this->get_parameter("frame_rate").as_double();
    loop_video_ = this->get_parameter("loop_video").as_bool();
    resize_ = this->get_parameter("resize").as_bool();
    resize_width_ = this->get_parameter("resize_width").as_int();
    resize_height_ = this->get_parameter("resize_height").as_int();

    // 初始化视频源
    if (source_mode_ == "video_file") {
      if (video_path_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Video path is empty!");
        rclcpp::shutdown();
        return;
      }
      cap_.open(video_path_);
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open video: %s", video_path_.c_str());
        rclcpp::shutdown();
        return;
      }
      double video_fps = cap_.get(cv::CAP_PROP_FPS);
      if (video_fps > 0) {
        frame_rate_ = video_fps;
        RCLCPP_INFO(this->get_logger(), "Using video FPS: %.2f", video_fps);
      }
    } else if (source_mode_ == "usb_camera") {
      cap_.open(camera_id_);
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open USB camera: %d", camera_id_);
        rclcpp::shutdown();
        return;
      }
      cap_.set(cv::CAP_PROP_FPS, frame_rate_);
      cap_.set(cv::CAP_PROP_FRAME_WIDTH, resize_width_);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, resize_height_);
      RCLCPP_INFO(this->get_logger(), "Opened USB camera %d", camera_id_);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Unknown source mode: %s", source_mode_.c_str());
      rclcpp::shutdown();
      return;
    }

    // 创建发布器
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    publisher_ = image_transport::create_publisher(this, "image_raw", qos.get_rmw_qos_profile());

    // 启动采集线程
    capture_thread_ = std::thread(&MultiSourceCameraNode::captureLoop, this);

    // 定时发布线程
    int interval_ms = static_cast<int>(1000.0 / frame_rate_);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(interval_ms),
      std::bind(&MultiSourceCameraNode::publishFrame, this)
    );

    RCLCPP_INFO(this->get_logger(),
      "MultiSourceCameraNode started - Mode: %s, FPS: %.1f",
      source_mode_.c_str(), frame_rate_);
  }

  ~MultiSourceCameraNode()
  {
    stop_capture_ = true;
    if (capture_thread_.joinable())
      capture_thread_.join();
  }

private:
  // 采集线程
  void captureLoop()
  {
    cv::Mat frame;
    auto last_time = std::chrono::steady_clock::now();

    while (rclcpp::ok() && !stop_capture_) {
      if (!cap_.read(frame) || frame.empty()) {
        if (source_mode_ == "video_file" && loop_video_) {
          cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
          continue;
        } else {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No frame captured");
          continue;
        }
      }

      if (resize_) {
        cv::resize(frame, frame, cv::Size(resize_width_, resize_height_));
      }

      {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = frame.clone();
      }

      // 控速：防止视频读取太快
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
      int delay = static_cast<int>(1000.0 / frame_rate_);
      if (elapsed < delay)
        std::this_thread::sleep_for(std::chrono::milliseconds(delay - elapsed));
      last_time = std::chrono::steady_clock::now();
    }
  }

  // 发布帧
  void publishFrame()
  {
    cv::Mat frame;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (latest_frame_.empty()) return;
      frame = latest_frame_.clone();
    }

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();

    // 时间戳策略
    if (source_mode_ == "video_file") {
      double msec = cap_.get(cv::CAP_PROP_POS_MSEC);
      msg->header.stamp = rclcpp::Time(static_cast<int64_t>(msec * 1e6));  // 视频帧时间
    } else {
      msg->header.stamp = this->now();  // 摄像头实时时间
    }

    msg->header.frame_id = "camera_optical_frame";
    publisher_.publish(msg);
  }

private:
  cv::VideoCapture cap_;
  image_transport::Publisher publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  // 参数
  std::string source_mode_;
  std::string video_path_;
  int camera_id_;
  double frame_rate_;
  bool loop_video_;
  bool resize_;
  int resize_width_;
  int resize_height_;

  // 状态
  std::thread capture_thread_;
  std::mutex frame_mutex_;
  cv::Mat latest_frame_;
  std::atomic<bool> stop_capture_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MultiSourceCameraNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
