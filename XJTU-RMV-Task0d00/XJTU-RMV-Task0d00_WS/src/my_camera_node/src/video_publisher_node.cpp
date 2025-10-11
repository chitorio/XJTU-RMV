#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class DualSourceVideoPublisher : public rclcpp::Node
{
public:
  DualSourceVideoPublisher() : Node("dual_source_video_publisher")
  {
    // 参数声明
    this->declare_parameter<std::string>("video_path", "");
    this->declare_parameter<int>("camera_id", -1);
    this->declare_parameter<double>("publish_fps", 30.0);
    this->declare_parameter<bool>("loop", true);
    this->declare_parameter<bool>("resize", true);
    this->declare_parameter<int>("resize_width", 640);
    this->declare_parameter<int>("resize_height", 480);
    this->declare_parameter<bool>("use_camera", false);

    // 获取参数
    video_path_ = this->get_parameter("video_path").as_string();
    camera_id_ = this->get_parameter("camera_id").as_int();
    publish_fps_ = this->get_parameter("publish_fps").as_double();
    loop_ = this->get_parameter("loop").as_bool();
    resize_ = this->get_parameter("resize").as_bool();
    resize_width_ = this->get_parameter("resize_width").as_int();
    resize_height_ = this->get_parameter("resize_height").as_int();
    use_camera_ = this->get_parameter("use_camera").as_bool();

    // 初始化视频源
    if (!initializeVideoSource()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to initialize video source!");
      rclcpp::shutdown();
      return;
    }

    // 高性能QoS配置
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
    qos.best_effort();
    
    // 创建发布器
    publisher_ = image_transport::create_publisher(this, "image_raw", qos.get_rmw_qos_profile());
    
    // 计算发布间隔
    int publish_interval_ms = static_cast<int>(1000.0 / publish_fps_);
    
    // 定时器
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(publish_interval_ms),
      std::bind(&DualSourceVideoPublisher::timerCallback, this));
      
    RCLCPP_INFO(this->get_logger(), 
                "Video Publisher started - Source: %s, FPS: %.1f, Resize: %dx%d",
                use_camera_ ? "Camera" : video_path_.c_str(),
                publish_fps_,
                resize_width_, resize_height_);
  }

private:
  bool initializeVideoSource()
  {
    if (use_camera_) {
      // 使用摄像头
      if (camera_id_ < 0) {
        RCLCPP_ERROR(this->get_logger(), "Invalid camera ID: %d", camera_id_);
        return false;
      }
      
      cap_.open(camera_id_);
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open camera: %d", camera_id_);
        return false;
      }
      
      // 设置摄像头参数
      cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
      cap_.set(cv::CAP_PROP_FPS, 30);
      
      RCLCPP_INFO(this->get_logger(), "Camera opened successfully: %d", camera_id_);
    } else {
      // 使用视频文件
      if (video_path_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Video path is empty!");
        return false;
      }
      
      cap_.open(video_path_);
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open video file: %s", video_path_.c_str());
        return false;
      }
      
      double video_fps = cap_.get(cv::CAP_PROP_FPS);
      RCLCPP_INFO(this->get_logger(), "Video FPS: %.2f", video_fps);
      
      // 如果视频帧率低于目标帧率，使用视频原始帧率
      if (video_fps > 0 && video_fps < publish_fps_) {
        publish_fps_ = video_fps;
        RCLCPP_WARN(this->get_logger(), "Video FPS (%.1f) is lower than target, using video FPS", video_fps);
      }
    }
    
    return true;
  }

  void timerCallback()
  {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cv::Mat frame;
    if (!cap_.read(frame)) {
      if (use_camera_) {
        RCLCPP_ERROR(this->get_logger(), "Failed to read frame from camera");
        return;
      } else {
        // 视频文件结束
        if (loop_) {
          cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
          if (!cap_.read(frame)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to restart video");
            return;
          }
          RCLCPP_DEBUG(this->get_logger(), "Video restarted");
        } else {
          RCLCPP_INFO(this->get_logger(), "End of video reached");
          rclcpp::shutdown();
          return;
        }
      }
    }

    // 调整图像大小
    if (resize_ && !frame.empty()) {
      cv::resize(frame, frame, cv::Size(resize_width_, resize_height_));
    }

    // 发布图像
    if (!frame.empty()) {
      try {
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        msg->header.stamp = this->now();
        msg->header.frame_id = "camera_optical_frame";
        
        publisher_.publish(msg);
        
        // 性能统计
        logPerformance(start_time);
        
      } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      }
    }
  }

  void logPerformance(const std::chrono::high_resolution_clock::time_point& start_time)
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    frame_count_++;
    total_publish_time_ += duration.count();
    
    if (frame_count_ >= 30) {
      double avg_time_ms = (total_publish_time_ / frame_count_) / 1000.0;
      double actual_fps = 1000000.0 / (total_publish_time_ / frame_count_);
      
      RCLCPP_INFO(this->get_logger(), 
                  "Publishing: %.1f FPS (target: %.1f), avg time: %.2f ms", 
                  actual_fps, publish_fps_, avg_time_ms);
      
      frame_count_ = 0;
      total_publish_time_ = 0;
    }
  }

  cv::VideoCapture cap_;
  image_transport::Publisher publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  
  // 参数
  std::string video_path_;
  int camera_id_;
  double publish_fps_;
  bool loop_;
  bool resize_;
  int resize_width_;
  int resize_height_;
  bool use_camera_;
  
  // 性能统计
  int frame_count_ = 0;
  long long total_publish_time_ = 0;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DualSourceVideoPublisher>());
  rclcpp::shutdown();
  return 0;
}