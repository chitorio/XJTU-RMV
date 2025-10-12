// multi_source_camera_node.cpp (硬件加速优化版)
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <thread>
#include <atomic>
#include <chrono>

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
    this->declare_parameter<bool>("use_thread", true);
    this->declare_parameter<bool>("use_hw_accel", true);  // 新增：硬件加速开关

    // 获取参数
    auto source_mode = this->get_parameter("source_mode").as_string();
    auto video_path = this->get_parameter("video_path").as_string();
    auto camera_id = this->get_parameter("camera_id").as_int();
    frame_rate_ = this->get_parameter("frame_rate").as_double();
    loop_video_ = this->get_parameter("loop_video").as_bool();
    resize_ = this->get_parameter("resize").as_bool();
    resize_width_ = this->get_parameter("resize_width").as_int();
    resize_height_ = this->get_parameter("resize_height").as_int();
    bool use_thread = this->get_parameter("use_thread").as_bool();
    bool use_hw_accel = this->get_parameter("use_hw_accel").as_bool();

    // 初始化视频源
    if (source_mode == "video_file") {
      if (video_path.empty()) throw std::runtime_error("Video path is empty!");
      
      // 尝试硬件加速
      if (use_hw_accel) {
        #if CV_VERSION_MAJOR >= 4
        cap_.open(video_path, cv::CAP_FFMPEG);
        cap_.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
        #else
        cap_.open(video_path);
        #endif
      } else {
        cap_.open(video_path);
      }
      
      if (!cap_.isOpened()) {
        RCLCPP_WARN(this->get_logger(), "Failed to open with hardware acceleration, trying software...");
        cap_.open(video_path);  // 回退到软件解码
      }
      
      if (!cap_.isOpened()) throw std::runtime_error("Failed to open video: " + video_path);
      
      // 获取视频实际帧率
      double video_fps = cap_.get(cv::CAP_PROP_FPS);
      if (video_fps > 0 && video_fps < frame_rate_) {
        frame_rate_ = video_fps;
        RCLCPP_WARN(this->get_logger(), "Using video FPS: %.1f", frame_rate_);
      }
      
      RCLCPP_INFO(this->get_logger(), "Video opened: %s, FPS: %.1f", video_path.c_str(), video_fps);
      
    } else if (source_mode == "usb_camera") {
      cap_.open(camera_id);
      if (!cap_.isOpened()) throw std::runtime_error("Failed to open USB camera: " + std::to_string(camera_id));
      
      // 设置USB摄像头参数
      cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
      cap_.set(cv::CAP_PROP_FPS, frame_rate_);
      cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
      
      // USB摄像头硬件加速
      if (use_hw_accel) {
        #if CV_VERSION_MAJOR >= 4
        cap_.set(cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY);
        #endif
      }
      
      RCLCPP_INFO(this->get_logger(), "USB Camera opened: %d", camera_id);
    } else {
      throw std::runtime_error("Unknown source mode: " + source_mode);
    }

    // 创建发布器
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    publisher_ = image_transport::create_publisher(this, "image_raw", qos.get_rmw_qos_profile());

    if (use_thread) {
      // 方法1：使用独立线程（最高性能）
      capture_thread_ = std::thread(&MultiSourceCameraNode::continuousCapture, this);
      RCLCPP_INFO(this->get_logger(), "Using threaded capture mode at %.1f FPS", frame_rate_);
    } else {
      // 方法2：使用定时器（兼容性更好）
      int interval_ms = static_cast<int>(1000.0 / frame_rate_);
      timer_ = this->create_wall_timer(
        std::chrono::milliseconds(interval_ms),
        std::bind(&MultiSourceCameraNode::captureAndPublish, this)
      );
      RCLCPP_INFO(this->get_logger(), "Using timer mode at %.1f FPS", frame_rate_);
    }

    // 性能监控定时器
    stats_timer_ = this->create_wall_timer(
      std::chrono::seconds(2),
      std::bind(&MultiSourceCameraNode::printStats, this)
    );
  }

  ~MultiSourceCameraNode() override
  {
    running_ = false;
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
  }

private:
  void continuousCapture()
  {
    auto frame_interval = std::chrono::microseconds(static_cast<int>(1000000.0 / frame_rate_));
    int consecutive_failures = 0;
    const int max_failures = 10;
    
    while (rclcpp::ok() && running_) {
      auto start_time = std::chrono::steady_clock::now();
      
      cv::Mat frame;
      if (!cap_.read(frame) || frame.empty()) {
        consecutive_failures++;
        
        if (loop_video_ && consecutive_failures < max_failures) {
          cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
          RCLCPP_DEBUG(this->get_logger(), "Video restarted");
          continue;
        } else {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                              "Failed to read frame (%d consecutive failures)", consecutive_failures);
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
      }
      
      consecutive_failures = 0;

      // 快速处理
      if (resize_ && (frame.cols != resize_width_ || frame.rows != resize_height_)) {
        cv::resize(frame, frame, cv::Size(resize_width_, resize_height_));
      }

      // 发布图像
      auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
      msg->header.stamp = this->now();
      msg->header.frame_id = "camera_optical_frame";
      publisher_.publish(std::move(msg));

      frame_count_++;
      
      // 精确控制帧率
      auto end_time = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      if (elapsed < frame_interval) {
        std::this_thread::sleep_for(frame_interval - elapsed);
      }
    }
  }

  void captureAndPublish()
  {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      if (loop_video_) {
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!cap_.read(frame) || frame.empty()) return;
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "End of video stream");
        return;
      }
    }

    if (resize_) {
      cv::resize(frame, frame, cv::Size(resize_width_, resize_height_));
    }

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = this->now();
    msg->header.frame_id = "camera_optical_frame";
    publisher_.publish(std::move(msg));

    frame_count_++;
  }

  void printStats()
  {
    double fps = frame_count_ / 2.0;
    RCLCPP_INFO(this->get_logger(), "Publishing at %.1f FPS", fps);
    frame_count_ = 0;
  }

private:
  cv::VideoCapture cap_;
  image_transport::Publisher publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr stats_timer_;
  std::thread capture_thread_;
  
  std::atomic<bool> running_{true};
  std::atomic<int> frame_count_{0};

  double frame_rate_;
  bool loop_video_;
  bool resize_;
  int resize_width_;
  int resize_height_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<MultiSourceCameraNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("rclcpp"), "Node creation failed: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}