#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class VideoPublisherNode : public rclcpp::Node
{
public:
  VideoPublisherNode() : Node("video_publisher_node")
  {
    // 声明一个参数，用于从命令行或launch文件接收视频路径
    this->declare_parameter<std::string>("video_path", "");

    // 获取视频路径参数
    std::string video_path = this->get_parameter("video_path").as_string();
    if (video_path.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Video path parameter is not set!");
      rclcpp::shutdown();
      return;
    }
    
    // 打开视频文件
    cap_.open(video_path);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open video file: %s", video_path.c_str());
      rclcpp::shutdown();
      return;
    }

    // 创建发布器
    publisher_ = image_transport::create_publisher(this, "image_raw");
    
    // 创建一个定时器，以视频的帧率来触发图像发布
    double fps = cap_.get(cv::CAP_PROP_FPS);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / fps)),
      std::bind(&VideoPublisherNode::timerCallback, this));
      
    RCLCPP_INFO(this->get_logger(), "Video Publisher started. Publishing from '%s' at %.1f FPS.", video_path.c_str(), fps);
  }

private:
  void timerCallback()
  {
    cv::Mat frame;
    cap_ >> frame; // 读取一帧

    if (frame.empty()) {
      RCLCPP_INFO(this->get_logger(), "End of video file. Restarting.");
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0); // 循环播放
      cap_ >> frame;
      if(frame.empty()) return; // 如果视频还是空的，则退出
    }

    cv::resize(frame, frame, cv::Size(640, 480));

    // 将OpenCV图像转换为ROS消息并发布
    sensor_msgs::msg::Image::SharedPtr msg = 
      cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    publisher_.publish(msg);
  }

  cv::VideoCapture cap_;
  image_transport::Publisher publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VideoPublisherNode>());
  rclcpp::shutdown();
  return 0;
}