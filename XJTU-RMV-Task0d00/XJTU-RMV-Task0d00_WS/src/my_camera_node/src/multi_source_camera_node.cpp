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

namespace multi_camera
{

class MultiSourceCameraNode : public rclcpp::Node
{
public:
  explicit MultiSourceCameraNode(const rclcpp::NodeOptions & options) : Node("multi_source_camera_node", options)
  {
    RCLCPP_INFO(this->get_logger(), "Starting MultiSourceCameraNode!");

    // 声明所有参数
    declareParameters();

    // 根据模式初始化
    std::string source_mode = this->get_parameter("source_mode").as_string();
    
    if (source_mode == "hik_camera") {
      initHikCamera();
    } else if (source_mode == "video_file") {
      initVideoFile();
    } else {
      RCLCPP_FATAL(this->get_logger(), "Unknown source mode: %s, use 'hik_camera' or 'video_file'", source_mode.c_str());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "MultiSourceCameraNode initialized successfully! Mode: %s", source_mode.c_str());
  }

  ~MultiSourceCameraNode() override
  {
    running_ = false;
    
    if (camera_manager_thread_.joinable()) {
      camera_manager_thread_.join();
    }
    
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    
    if (source_mode_ == "hik_camera") {
      closeHikCamera();
    } else if (source_mode_ == "video_file") {
      closeVideoCamera();
    }
    
    RCLCPP_INFO(this->get_logger(), "MultiSourceCameraNode destroyed!");
  }

private:
  void declareParameters()
  {
    // 视频源模式选择
    this->declare_parameter("source_mode", "video_file"); // hik_camera, video_file
    
    // 海康相机参数
    this->declare_parameter("camera_serial", "");
    this->declare_parameter("camera_ip", "");
    
    // 视频文件参数
    this->declare_parameter("video_path", "");
    this->declare_parameter("loop_video", true);
    
    // 通用发布设置
    this->declare_parameter("frame_id", "camera_optical_frame");
    this->declare_parameter("image_topic", "image_raw");
    this->declare_parameter("use_sensor_data_qos", true);
    this->declare_parameter("camera_name", "camera");
    this->declare_parameter("camera_info_url", "");
    
    // 采集参数
    this->declare_parameter("frame_rate", 30.0);
    this->declare_parameter("width", 640);
    this->declare_parameter("height", 480);
    this->declare_parameter("pixel_format", "bgr8");
    
    // 海康相机控制参数
    this->declare_parameter("exposure_time", 10000.0);
    this->declare_parameter("gain", 0.0);
    this->declare_parameter("auto_exposure", false);
    this->declare_parameter("auto_gain", false);
    
    // 重连设置
    this->declare_parameter("reconnect_interval", 2.0);
    this->declare_parameter("max_reconnect_attempts", 0);
  }

  void initHikCamera()
  {
    source_mode_ = "hik_camera";
    
    // 初始化SDK
    if (MV_CC_Initialize() != MV_OK) {
      RCLCPP_FATAL(this->get_logger(), "Failed to initialize Hikvision SDK!");
      return;
    }

    // 启动相机管理线程
    camera_manager_thread_ = std::thread(&MultiSourceCameraNode::hikCameraManager, this);
  }

  void initVideoFile()
  {
    source_mode_ = "video_file";
    std::string video_path = this->get_parameter("video_path").as_string();
    loop_video_ = this->get_parameter("loop_video").as_bool();
    
    if (video_path.empty()) {
      RCLCPP_FATAL(this->get_logger(), "Video path is empty!");
      return;
    }
    
    video_cap_.open(video_path);
    if (!video_cap_.isOpened()) {
      RCLCPP_FATAL(this->get_logger(), "Failed to open video file: %s", video_path.c_str());
      return;
    }
    
    double video_fps = video_cap_.get(cv::CAP_PROP_FPS);
    RCLCPP_INFO(this->get_logger(), "Video opened: %s, FPS: %.2f", video_path.c_str(), video_fps);
    
    // 直接启动采集线程
    startCaptureThread();
  }

  void hikCameraManager()
  {
    int reconnect_attempts = 0;
    
    while (rclcpp::ok() && running_) {
      if (!camera_connected_) {
        RCLCPP_INFO(this->get_logger(), "Attempting to connect to Hik camera...");
        
        if (connectToHikCamera()) {
          if (configureHikCamera() && startHikGrabbing()) {
            camera_connected_ = true;
            reconnect_attempts = 0;
            startCaptureThread();
            RCLCPP_INFO(this->get_logger(), "Hik camera connected and started successfully!");
          } else {
            closeHikCamera();
          }
        } else {
          reconnect_attempts++;
          RCLCPP_WARN(this->get_logger(), "Failed to connect to Hik camera (attempt %d)", reconnect_attempts);
          
          double reconnect_interval = this->get_parameter("reconnect_interval").as_double();
          int max_attempts = this->get_parameter("max_reconnect_attempts").as_int();
          
          if (max_attempts > 0 && reconnect_attempts >= max_attempts) {
            RCLCPP_FATAL(this->get_logger(), "Max reconnect attempts (%d) reached!", max_attempts);
            rclcpp::shutdown();
            break;
          }
          
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

    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
    if (ret != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "Enum devices failed! ret[0x%x]", ret);
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "Found %d camera(s)", device_list.nDeviceNum);

    if (device_list.nDeviceNum == 0) {
      RCLCPP_ERROR(this->get_logger(), "No cameras found!");
      return false;
    }

    std::string target_serial = this->get_parameter("camera_serial").as_string();
    std::string target_ip = this->get_parameter("camera_ip").as_string();

    int target_index = -1;
    for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
      MV_CC_DEVICE_INFO* device_info = device_list.pDeviceInfo[i];
      
      if (!target_ip.empty() && device_info->nTLayerType == MV_GIGE_DEVICE) {
        char current_ip[16] = {0};
        unsigned int ip_val = device_info->SpecialInfo.stGigEInfo.nCurrentIp;
        snprintf(current_ip, sizeof(current_ip), "%d.%d.%d.%d",
                (ip_val >> 24) & 0xff, (ip_val >> 16) & 0xff,
                (ip_val >> 8) & 0xff, ip_val & 0xff);
        
        std::string current_ip_str(current_ip);
        if (target_ip == current_ip_str) {
          target_index = i;
          RCLCPP_INFO(this->get_logger(), "Found camera by IP: %s", current_ip);
          break;
        }
      }
      
      if (!target_serial.empty()) {
        std::string serial_number;
        if (device_info->nTLayerType == MV_GIGE_DEVICE) {
          serial_number = std::string(reinterpret_cast<const char*>(device_info->SpecialInfo.stGigEInfo.chSerialNumber));
        } else if (device_info->nTLayerType == MV_USB_DEVICE) {
          serial_number = std::string(reinterpret_cast<const char*>(device_info->SpecialInfo.stUsb3VInfo.chSerialNumber));
        }
        
        if (target_serial == serial_number) {
          target_index = i;
          RCLCPP_INFO(this->get_logger(), "Found camera by serial: %s", serial_number.c_str());
          break;
        }
      }
    }

    if (target_index == -1) {
      target_index = 0;
      RCLCPP_INFO(this->get_logger(), "Using first available camera");
    }

    int ret_handle = MV_CC_CreateHandle(&hik_camera_handle_, device_list.pDeviceInfo[target_index]);
    if (ret_handle != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "Create handle failed! ret[0x%x]", ret_handle);
      return false;
    }

    int ret_open = MV_CC_OpenDevice(hik_camera_handle_);
    if (ret_open != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "Open device failed! ret[0x%x]", ret_open);
      MV_CC_DestroyHandle(hik_camera_handle_);
      hik_camera_handle_ = nullptr;
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "Hik camera connected successfully");
    return true;
  }

  bool configureHikCamera()
  {
    int ret = MV_CC_SetEnumValue(hik_camera_handle_, "TriggerMode", 0);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set trigger mode failed! ret[0x%x]", ret);
    }

    double frame_rate = this->get_parameter("frame_rate").as_double();
    ret = MV_CC_SetFloatValue(hik_camera_handle_, "AcquisitionFrameRate", frame_rate);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set frame rate failed! ret[0x%x]", ret);
    }

    int width = this->get_parameter("width").as_int();
    int height = this->get_parameter("height").as_int();
    if (width > 0 && height > 0) {
      int ret_width = MV_CC_SetIntValueEx(hik_camera_handle_, "Width", static_cast<int64_t>(width));
      int ret_height = MV_CC_SetIntValueEx(hik_camera_handle_, "Height", static_cast<int64_t>(height));
      if (ret_width != MV_OK || ret_height != MV_OK) {
        RCLCPP_WARN(this->get_logger(), "Set image size failed, using default size");
      }
    }

    bool auto_exposure = this->get_parameter("auto_exposure").as_bool();
    ret = MV_CC_SetEnumValue(hik_camera_handle_, "ExposureAuto", auto_exposure ? 1 : 0);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set exposure auto failed! ret[0x%x]", ret);
    }

    if (!auto_exposure) {
      double exposure_time = this->get_parameter("exposure_time").as_double();
      ret = MV_CC_SetFloatValue(hik_camera_handle_, "ExposureTime", exposure_time);
      if (ret != MV_OK) {
        RCLCPP_WARN(this->get_logger(), "Set exposure time failed! ret[0x%x]", ret);
      }
    }

    bool auto_gain = this->get_parameter("auto_gain").as_bool();
    ret = MV_CC_SetEnumValue(hik_camera_handle_, "GainAuto", auto_gain ? 1 : 0);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set gain auto failed! ret[0x%x]", ret);
    }

    if (!auto_gain) {
      double gain = this->get_parameter("gain").as_double();
      ret = MV_CC_SetFloatValue(hik_camera_handle_, "Gain", gain);
      if (ret != MV_OK) {
        RCLCPP_WARN(this->get_logger(), "Set gain failed! ret[0x%x]", ret);
      }
    }

    std::string pixel_format = this->get_parameter("pixel_format").as_string();
    unsigned int pixel_format_enum = PixelType_Gvsp_BGR8_Packed;
    
    if (pixel_format == "bgr8") {
      pixel_format_enum = PixelType_Gvsp_BGR8_Packed;
    } else if (pixel_format == "rgb8") {
      pixel_format_enum = PixelType_Gvsp_RGB8_Packed;
    } else if (pixel_format == "mono8") {
      pixel_format_enum = PixelType_Gvsp_Mono8;
    } else {
      RCLCPP_WARN(this->get_logger(), "Unsupported pixel format: %s, using bgr8", pixel_format.c_str());
    }

    ret = MV_CC_SetEnumValue(hik_camera_handle_, "PixelFormat", pixel_format_enum);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set pixel format failed! ret[0x%x]", ret);
    }

    RCLCPP_INFO(this->get_logger(), "Hik camera configured successfully");
    return true;
  }

  bool startHikGrabbing()
  {
    int ret = MV_CC_StartGrabbing(hik_camera_handle_);
    if (ret != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "Start grabbing failed! ret[0x%x]", ret);
      return false;
    }

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
    MV_CC_Finalize();
  }

  void closeVideoCamera()
  {
    if (video_cap_.isOpened()) {
      video_cap_.release();
    }
  }

  void startCaptureThread()
  {
    // 初始化发布器
    bool use_sensor_data_qos = this->get_parameter("use_sensor_data_qos").as_bool();
    rclcpp::QoS qos_profile = use_sensor_data_qos ? rclcpp::SensorDataQoS() : rclcpp::QoS(10);
    qos_profile.reliable();

    std::string image_topic = this->get_parameter("image_topic").as_string();
    camera_pub_ = image_transport::create_camera_publisher(this, image_topic, qos_profile.get_rmw_qos_profile());

    // 初始化相机信息管理器
    std::string camera_name = this->get_parameter("camera_name").as_string();
    std::string camera_info_url = this->get_parameter("camera_info_url").as_string();
    camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name);

    if (!camera_info_url.empty() && camera_info_manager_->validateURL(camera_info_url)) {
      camera_info_manager_->loadCameraInfo(camera_info_url);
      RCLCPP_INFO(this->get_logger(), "Loaded camera info from: %s", camera_info_url.c_str());
    }

    // 启动采集线程
    capture_thread_ = std::thread(&MultiSourceCameraNode::captureImages, this);
  }

  void captureImages()
  {
    RCLCPP_INFO(this->get_logger(), "Starting image capture thread for mode: %s", source_mode_.c_str());

    auto last_stat_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    const int stat_interval_frames = 100;

    while (rclcpp::ok() && running_) {
      cv::Mat frame;
      bool frame_ok = false;

      if (source_mode_ == "hik_camera" && camera_connected_) {
        frame_ok = getHikFrame(frame);
      } else if (source_mode_ == "video_file") {
        frame_ok = getVideoFrame(frame);
      }

      if (frame_ok && !frame.empty()) {
        publishFrame(frame);
        
        // 帧率统计
        frame_count++;
        if (frame_count >= stat_interval_frames) {
          auto current_time = std::chrono::steady_clock::now();
          auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_stat_time).count();
          double actual_fps = (frame_count * 1000.0) / elapsed_time;
          RCLCPP_INFO(this->get_logger(), "Current frame rate: %.1f Hz", actual_fps);
          frame_count = 0;
          last_stat_time = current_time;
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    
    RCLCPP_INFO(this->get_logger(), "Image capture thread exiting");
  }

  bool getHikFrame(cv::Mat& frame)
  {
    MV_FRAME_OUT out_frame;
    int ret = MV_CC_GetImageBuffer(hik_camera_handle_, &out_frame, 1000);
    
    if (ret == MV_OK) {
      // 转换图像格式
      cv::Mat temp_frame(out_frame.stFrameInfo.nHeight, out_frame.stFrameInfo.nWidth, CV_8UC3);
      
      MV_CC_PIXEL_CONVERT_PARAM convert_param;
      memset(&convert_param, 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM));
      convert_param.nWidth = out_frame.stFrameInfo.nWidth;
      convert_param.nHeight = out_frame.stFrameInfo.nHeight;
      convert_param.pSrcData = out_frame.pBufAddr;
      convert_param.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
      convert_param.enSrcPixelType = out_frame.stFrameInfo.enPixelType;
      convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
      convert_param.pDstBuffer = temp_frame.data;
      convert_param.nDstBufferSize = temp_frame.total() * temp_frame.elemSize();

      int convert_ret = MV_CC_ConvertPixelType(hik_camera_handle_, &convert_param);
      MV_CC_FreeImageBuffer(hik_camera_handle_, &out_frame);

      if (convert_ret == MV_OK) {
        frame = temp_frame.clone();
        return true;
      }
    }
    
    return false;
  }

  bool getVideoFrame(cv::Mat& frame)
  {
    if (!video_cap_.read(frame)) {
      if (loop_video_) {
        video_cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        if (!video_cap_.read(frame)) {
          RCLCPP_ERROR(this->get_logger(), "Failed to restart video");
          return false;
        }
      } else {
        RCLCPP_INFO(this->get_logger(), "End of video reached");
        return false;
      }
    }
    return true;
  }

  void publishFrame(const cv::Mat& frame)
  {
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = this->now();
    msg->header.frame_id = this->get_parameter("frame_id").as_string();

    auto camera_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>(camera_info_manager_->getCameraInfo());
    camera_info_msg->header = msg->header;
    
    camera_pub_.publish(msg, camera_info_msg);
  }

  // 成员变量
  std::string source_mode_;
  void * hik_camera_handle_ = nullptr;
  cv::VideoCapture video_cap_;
  std::atomic<bool> running_{true};
  std::atomic<bool> camera_connected_{false};
  bool loop_video_ = true;
  
  std::thread camera_manager_thread_;
  std::thread capture_thread_;
  
  image_transport::CameraPublisher camera_pub_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
};

}  // namespace multi_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(multi_camera::MultiSourceCameraNode)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  // 添加 NodeOptions 参数
  auto node = std::make_shared<multi_camera::MultiSourceCameraNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}