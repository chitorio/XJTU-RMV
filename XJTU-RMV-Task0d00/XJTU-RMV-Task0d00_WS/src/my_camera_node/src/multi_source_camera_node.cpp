#include "MvCameraControl.h"
// ROS
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>

namespace multi_camera
{

class MultiSourceCameraNode : public rclcpp::Node
{
public:
  explicit MultiSourceCameraNode(const rclcpp::NodeOptions & options) : Node("multi_source_camera_node", options)
  {
    RCLCPP_INFO(this->get_logger(), "Starting MultiSourceCameraNode!");
    declareParameters();
    source_mode_ = this->get_parameter("source_mode").as_string();
    if (source_mode_ == "hik_camera") initHikCamera();
    else if (source_mode_ == "video_file") initVideoFile();
    else throw std::runtime_error("Unknown source mode: " + source_mode_);
    RCLCPP_INFO(this->get_logger(), "MultiSourceCameraNode initialized successfully! Mode: %s", source_mode_.c_str());
  }

  ~MultiSourceCameraNode() override
  {
    running_ = false;
    if (capture_thread_.joinable()) capture_thread_.join();
    if (camera_manager_thread_.joinable()) camera_manager_thread_.join();
    if (source_mode_ == "hik_camera") closeHikCamera();
    else if (source_mode_ == "video_file") closeVideoCamera();
    RCLCPP_INFO(this->get_logger(), "MultiSourceCameraNode destroyed!");
  }

private:
  void declareParameters()
  {
    this->declare_parameter("source_mode", "video_file");
    this->declare_parameter("video_path", "");
    this->declare_parameter("loop_video", true);
    this->declare_parameter("camera_serial", "");
    this->declare_parameter("camera_ip", "");
    this->declare_parameter("frame_id", "camera_optical_frame");
    this->declare_parameter("image_topic", "image_raw");
    this->declare_parameter("use_sensor_data_qos", true);
    this->declare_parameter("camera_name", "camera");
    this->declare_parameter("camera_info_url", "");
    this->declare_parameter("frame_rate", 120.0);
    this->declare_parameter("resize", false);
    this->declare_parameter("width", 640);
    this->declare_parameter("height", 480);
    this->declare_parameter("pixel_format", "bgr8");
    this->declare_parameter("exposure_time", 10000.0);
    this->declare_parameter("gain", 10.0);
    this->declare_parameter("auto_exposure", false);
    this->declare_parameter("auto_gain", false);
    this->declare_parameter("reconnect_interval", 2.0);
    this->declare_parameter("max_reconnect_attempts", 0);
  }

  void initHikCamera()
  {
    if (MV_CC_Initialize() != MV_OK) throw std::runtime_error("Failed to initialize Hikvision SDK!");
    camera_manager_thread_ = std::thread(&MultiSourceCameraNode::hikCameraManager, this);
  }

  void initVideoFile()
  {
    std::string video_path = this->get_parameter("video_path").as_string();
    if (video_path.empty()) throw std::runtime_error("Video path is empty!");
    video_cap_.open(video_path);
    if (!video_cap_.isOpened()) throw std::runtime_error("Failed to open video file: " + video_path);
    startCaptureThread();
  }

  void hikCameraManager()
  {
    while (rclcpp::ok() && running_) {
      if (!camera_connected_) {
        RCLCPP_INFO(this->get_logger(), "Attempting to connect to Hik camera...");
        if (connectToHikCamera()) {
          if (configureHikCamera() && startHikGrabbing()) {
            camera_connected_ = true;
            startCaptureThread();
            RCLCPP_INFO(this->get_logger(), "Hik camera connected and started successfully!");
          } else {
            closeHikCamera();
          }
        } else {
          double reconnect_interval = this->get_parameter("reconnect_interval").as_double();
          std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(reconnect_interval * 1000)));
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }

  bool connectToHikCamera()
  {
    MV_CC_DEVICE_INFO_LIST device_list;
    memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
    if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list) != MV_OK || device_list.nDeviceNum == 0) {
        RCLCPP_WARN(this->get_logger(), "No Hik cameras found.");
        return false;
    }
    int device_idx = 0;
    if (MV_CC_CreateHandle(&hik_camera_handle_, device_list.pDeviceInfo[device_idx]) != MV_OK) return false;
    if (MV_CC_OpenDevice(hik_camera_handle_) != MV_OK) {
      MV_CC_DestroyHandle(hik_camera_handle_);
      hik_camera_handle_ = nullptr;
      return false;
    }
    if (device_list.pDeviceInfo[device_idx]->nTLayerType == MV_GIGE_DEVICE) {
      int nPacketSize = MV_CC_GetOptimalPacketSize(hik_camera_handle_);
      if (nPacketSize > 0) {
        if (MV_CC_SetIntValueEx(hik_camera_handle_, "GevSCPSPacketSize", nPacketSize) == MV_OK) {
          RCLCPP_INFO(this->get_logger(), "Optimal packet size set to %d", nPacketSize);
        }
      }
    }
    return true;
  }

  bool configureHikCamera()
  {
    double exposure_time = this->get_parameter("exposure_time").as_double();
    double gain = this->get_parameter("gain").as_double();
    
    // 让相机使用默认的最大分辨率进行采集，以获取完整视野
    MV_CC_SetEnumValue(hik_camera_handle_, "TriggerMode", 0);
    MV_CC_SetEnumValue(hik_camera_handle_, "ExposureAuto", this->get_parameter("auto_exposure").as_bool() ? 1 : 0);
    MV_CC_SetFloatValue(hik_camera_handle_, "ExposureTime", exposure_time);
    MV_CC_SetEnumValue(hik_camera_handle_, "GainAuto", this->get_parameter("auto_gain").as_bool() ? 1 : 0);
    MV_CC_SetFloatValue(hik_camera_handle_, "Gain", gain);
    
    RCLCPP_INFO(this->get_logger(), "Hik camera configured with Exposure: %.1f, Gain: %.1f. Capture resolution is native.", exposure_time, gain);
    return true;
  }

  bool startHikGrabbing()
  {
    if (MV_CC_StartGrabbing(hik_camera_handle_) != MV_OK) return false;
    RCLCPP_INFO(this->get_logger(), "Hik camera grabbing started");
    return true;
  }

  void closeHikCamera()
  {
    if (hik_camera_handle_) {
      MV_CC_StopGrabbing(hik_camera_handle_);
      MV_CC_CloseDevice(hik_camera_handle_);
      MV_CC_DestroyHandle(hik_camera_handle_);
      hik_camera_handle_ = nullptr;
    }
    camera_connected_ = false;
    if (source_mode_ == "hik_camera") MV_CC_Finalize();
  }

  void closeVideoCamera()
  {
    if (video_cap_.isOpened()) video_cap_.release();
  }

  void startCaptureThread()
  {
    // 创建图像发布器
    camera_pub_ = image_transport::create_camera_publisher(this, this->get_parameter("image_topic").as_string(), rclcpp::SensorDataQoS().get_rmw_qos_profile());
    
    // 创建camera_info发布器 - 新增！
    camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS());
    
    // 初始化相机信息管理器
    camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, this->get_parameter("camera_name").as_string(), this->get_parameter("camera_info_url").as_string());
    
    if(camera_info_manager_->loadCameraInfo(this->get_parameter("camera_info_url").as_string())) {
        RCLCPP_INFO(this->get_logger(), "Camera info loaded successfully.");
    } else {
        RCLCPP_WARN(this->get_logger(), "Failed to load camera info from: %s", this->get_parameter("camera_info_url").as_string().c_str());
    }
    
    capture_thread_ = std::thread(&MultiSourceCameraNode::captureImages, this);
  }

  void captureImages()
  {
    while (rclcpp::ok() && running_) {
      try {
        cv::Mat frame;
        bool frame_ok = (source_mode_ == "hik_camera" && camera_connected_) ? getHikFrame(frame) : getVideoFrame(frame);
        if (frame_ok && !frame.empty()) {
          publishFrame(frame);
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in capture thread: %s", e.what());
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
  }

  bool getHikFrame(cv::Mat& frame)
  {
    MV_FRAME_OUT out_frame;
    if (MV_CC_GetImageBuffer(hik_camera_handle_, &out_frame, 1000) != MV_OK) return false;
    
    cv::Mat temp_frame(out_frame.stFrameInfo.nHeight, out_frame.stFrameInfo.nWidth, CV_8UC3);
    MV_CC_PIXEL_CONVERT_PARAM cvt_param = {0};
    cvt_param.nWidth = out_frame.stFrameInfo.nWidth;
    cvt_param.nHeight = out_frame.stFrameInfo.nHeight;
    cvt_param.pSrcData = out_frame.pBufAddr;
    cvt_param.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
    cvt_param.enSrcPixelType = out_frame.stFrameInfo.enPixelType;
    cvt_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
    cvt_param.pDstBuffer = temp_frame.data;
    cvt_param.nDstBufferSize = temp_frame.total() * temp_frame.elemSize();

    bool success = (MV_CC_ConvertPixelType(hik_camera_handle_, &cvt_param) == MV_OK);
    MV_CC_FreeImageBuffer(hik_camera_handle_, &out_frame);
    if (success) frame = temp_frame;
    return success;
  }

  bool getVideoFrame(cv::Mat& frame)
  {
    if (!video_cap_.read(frame)) {
      if (this->get_parameter("loop_video").as_bool()) {
        video_cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        return video_cap_.read(frame);
      }
      return false;
    }
    return true;
  }

  void publishFrame(const cv::Mat& frame_const)
  {
    cv::Mat frame = frame_const.clone();
    if (this->get_parameter("resize").as_bool()) {
      cv::resize(frame, frame, cv::Size(this->get_parameter("width").as_int(), this->get_parameter("height").as_int()));
    }
    
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = this->now();
    msg->header.frame_id = this->get_parameter("frame_id").as_string();
    
    // 获取camera_info并发布 - 新增！
    auto camera_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>(camera_info_manager_->getCameraInfo());
    camera_info_msg->header = msg->header;
    
    // 发布图像和camera_info
    camera_pub_.publish(std::move(msg), camera_info_msg);
    
    // 单独发布camera_info话题 - 新增！
    camera_info_pub_->publish(*camera_info_msg);
  }

  std::string source_mode_;
  void* hik_camera_handle_ = nullptr;
  cv::VideoCapture video_cap_;
  std::atomic<bool> running_{true};
  std::atomic<bool> camera_connected_{false};
  std::thread camera_manager_thread_;
  std::thread capture_thread_;
  image_transport::CameraPublisher camera_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_; // 新增！
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
};

} // namespace multi_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(multi_camera::MultiSourceCameraNode)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<multi_camera::MultiSourceCameraNode>(rclcpp::NodeOptions());
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("rclcpp"), "Node creation failed: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}