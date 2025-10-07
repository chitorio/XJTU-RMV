#include "MvCameraControl.h"
// ROS
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace hik_camera
{

class HikCameraNode : public rclcpp::Node
{
public:
  explicit HikCameraNode(const rclcpp::NodeOptions & options) : Node("hik_camera_node", options)
  {
    RCLCPP_INFO(this->get_logger(), "Starting HikCameraNode!");

    // 声明所有参数
    declareParameters();

    // 初始化SDK
    if (MV_CC_Initialize() != MV_OK) {
      RCLCPP_FATAL(this->get_logger(), "Failed to initialize Hikvision SDK!");
      return;
    }

    // 启动相机管理线程
    camera_manager_thread_ = std::thread(&HikCameraNode::cameraManager, this);

    RCLCPP_INFO(this->get_logger(), "HikCameraNode initialized successfully!");
  }

  ~HikCameraNode() override
  {
    running_ = false;
    
    if (camera_manager_thread_.joinable()) {
      camera_manager_thread_.join();
    }
    
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    
    closeCamera();
    MV_CC_Finalize();
    
    RCLCPP_INFO(this->get_logger(), "HikCameraNode destroyed!");
  }

private:
  void declareParameters()
  {
    // 相机识别参数
    this->declare_parameter("camera_serial", "");
    this->declare_parameter("camera_ip", "");
    
    // 发布设置
    this->declare_parameter("frame_id", "camera_optical_frame");
    this->declare_parameter("image_topic", "image_raw");
    this->declare_parameter("use_sensor_data_qos", true);
    this->declare_parameter("camera_name", "hik_camera");
    this->declare_parameter("camera_info_url", "");
    
    // 采集参数
    this->declare_parameter("frame_rate", 30.0);
    this->declare_parameter("width", 0);
    this->declare_parameter("height", 0);
    this->declare_parameter("pixel_format", "bgr8");
    
    // 相机控制参数
    this->declare_parameter("exposure_time", 10000.0);
    this->declare_parameter("gain", 0.0);
    this->declare_parameter("auto_exposure", false);
    this->declare_parameter("auto_gain", false);
    
    // 重连设置
    this->declare_parameter("reconnect_interval", 2.0);
    this->declare_parameter("max_reconnect_attempts", 0);
    
    // 注册参数回调
    params_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&HikCameraNode::parametersCallback, this, std::placeholders::_1));
  }

  void cameraManager()
  {
    int reconnect_attempts = 0;
    
    while (rclcpp::ok() && running_) {
      if (!camera_connected_) {
        RCLCPP_INFO(this->get_logger(), "Attempting to connect to camera...");
        
        if (connectToCamera()) {
          if (configureCamera() && startGrabbing()) {
            camera_connected_ = true;
            reconnect_attempts = 0;
            startCaptureThread();
            RCLCPP_INFO(this->get_logger(), "Camera connected and started successfully!");
          } else {
            closeCamera();
          }
        } else {
          reconnect_attempts++;
          RCLCPP_WARN(this->get_logger(), "Failed to connect to camera (attempt %d)", reconnect_attempts);
          
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

  bool connectToCamera()
  {
    MV_CC_DEVICE_INFO_LIST device_list;
    memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备（支持GigE和USB）
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

    // 获取配置参数
    std::string target_serial = this->get_parameter("camera_serial").as_string();
    std::string target_ip = this->get_parameter("camera_ip").as_string();

    // 查找指定相机
    int target_index = -1;
    for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
      MV_CC_DEVICE_INFO* device_info = device_list.pDeviceInfo[i];
      
      // 通过IP匹配（GigE设备）
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
      
      // 通过序列号匹配
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

    // 如果没有指定相机，使用第一个可用相机
    if (target_index == -1) {
      target_index = 0;
      RCLCPP_INFO(this->get_logger(), "Using first available camera");
    }

    // 创建设备句柄
    int ret_handle = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[target_index]);
    if (ret_handle != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "Create handle failed! ret[0x%x]", ret_handle);
      return false;
    }

    // 打开设备
    int ret_open = MV_CC_OpenDevice(camera_handle_);
    if (ret_open != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "Open device failed! ret[0x%x]", ret_open);
      MV_CC_DestroyHandle(camera_handle_);
      camera_handle_ = nullptr;
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "Camera connected successfully");
    return true;
  }

  bool configureCamera()
  {
    // 设置触发模式为连续采集
    int ret = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", 0);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set trigger mode failed! ret[0x%x]", ret);
    }

    // 设置帧率
    double frame_rate = this->get_parameter("frame_rate").as_double();
    ret = MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", frame_rate);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set frame rate failed! ret[0x%x]", ret);
    }

    // 设置图像尺寸
    int width = this->get_parameter("width").as_int();
    int height = this->get_parameter("height").as_int();
    if (width > 0 && height > 0) {
      int ret_width = MV_CC_SetIntValueEx(camera_handle_, "Width", static_cast<int64_t>(width));
      int ret_height = MV_CC_SetIntValueEx(camera_handle_, "Height", static_cast<int64_t>(height));
      if (ret_width != MV_OK || ret_height != MV_OK) {
        RCLCPP_WARN(this->get_logger(), "Set image size failed, using default size");
      }
    }

    // 设置曝光模式
    bool auto_exposure = this->get_parameter("auto_exposure").as_bool();
    ret = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", auto_exposure ? 1 : 0);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set exposure auto failed! ret[0x%x]", ret);
    }

    if (!auto_exposure) {
      double exposure_time = this->get_parameter("exposure_time").as_double();
      ret = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_time);
      if (ret != MV_OK) {
        RCLCPP_WARN(this->get_logger(), "Set exposure time failed! ret[0x%x]", ret);
      }
    }

    // 设置增益模式
    bool auto_gain = this->get_parameter("auto_gain").as_bool();
    ret = MV_CC_SetEnumValue(camera_handle_, "GainAuto", auto_gain ? 1 : 0);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set gain auto failed! ret[0x%x]", ret);
    }

    if (!auto_gain) {
      double gain = this->get_parameter("gain").as_double();
      ret = MV_CC_SetFloatValue(camera_handle_, "Gain", gain);
      if (ret != MV_OK) {
        RCLCPP_WARN(this->get_logger(), "Set gain failed! ret[0x%x]", ret);
      }
    }

    // 设置像素格式
    std::string pixel_format = this->get_parameter("pixel_format").as_string();
    unsigned int pixel_format_enum = PixelType_Gvsp_BGR8_Packed; // 默认
    
    if (pixel_format == "bgr8") {
      pixel_format_enum = PixelType_Gvsp_BGR8_Packed;
    } else if (pixel_format == "rgb8") {
      pixel_format_enum = PixelType_Gvsp_RGB8_Packed;
    } else if (pixel_format == "mono8") {
      pixel_format_enum = PixelType_Gvsp_Mono8;
    } else {
      RCLCPP_WARN(this->get_logger(), "Unsupported pixel format: %s, using bgr8", pixel_format.c_str());
    }

    ret = MV_CC_SetEnumValue(camera_handle_, "PixelFormat", pixel_format_enum);
    if (ret != MV_OK) {
      RCLCPP_WARN(this->get_logger(), "Set pixel format failed! ret[0x%x]", ret);
    }

    RCLCPP_INFO(this->get_logger(), "Camera configured successfully");
    return true;
  }

  bool startGrabbing()
  {
    int ret = MV_CC_StartGrabbing(camera_handle_);
    if (ret != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "Start grabbing failed! ret[0x%x]", ret);
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "Image grabbing started");
    return true;
  }

  void stopGrabbing()
  {
    if (camera_handle_) {
      MV_CC_StopGrabbing(camera_handle_);
    }
  }

  void closeCamera()
  {
    if (camera_handle_) {
      stopGrabbing();
      MV_CC_CloseDevice(camera_handle_);
      MV_CC_DestroyHandle(camera_handle_);
      camera_handle_ = nullptr;
    }
    camera_connected_ = false;
  }

  void startCaptureThread()
  {
    // 初始化图像发布器（在连接成功后）
    bool use_sensor_data_qos = this->get_parameter("use_sensor_data_qos").as_bool();

    // 修复QoS配置 - 确保可靠性
    rclcpp::QoS qos_profile = use_sensor_data_qos ? 
                            rclcpp::SensorDataQoS() : 
                            rclcpp::QoS(10);
    qos_profile.reliable();  // 确保可靠性策略匹配

    std::string image_topic = this->get_parameter("image_topic").as_string();

    // 使用新的QoS配置创建发布器
    camera_pub_ = image_transport::create_camera_publisher(
        this, 
        image_topic, 
        qos_profile.get_rmw_qos_profile()
    );

    // 初始化相机信息管理器
    std::string camera_name = this->get_parameter("camera_name").as_string();
    std::string camera_info_url = this->get_parameter("camera_info_url").as_string();
    camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name);

    if (!camera_info_url.empty() && camera_info_manager_->validateURL(camera_info_url)) {
        camera_info_manager_->loadCameraInfo(camera_info_url);
        RCLCPP_INFO(this->get_logger(), "Loaded camera info from: %s", camera_info_url.c_str());
    } else {
        RCLCPP_WARN(this->get_logger(), "Using default camera info");
    }

    // 启动采集线程
    capture_thread_ = std::thread(&HikCameraNode::captureImages, this);
  }

  void captureImages()
  {
    MV_FRAME_OUT out_frame;
    int fail_count = 0;
    const int max_fail_count = 5;

    RCLCPP_INFO(this->get_logger(), "Starting image capture thread");

    // 初始化图像消息
    sensor_msgs::msg::Image image_msg;
    std::string frame_id = this->get_parameter("frame_id").as_string();
    image_msg.header.frame_id = frame_id;

    // 初始化像素转换参数
    MV_CC_PIXEL_CONVERT_PARAM convert_param;
    memset(&convert_param, 0, sizeof(MV_CC_PIXEL_CONVERT_PARAM));

    while (rclcpp::ok() && running_ && camera_connected_) {
      int ret = MV_CC_GetImageBuffer(camera_handle_, &out_frame, 1000);
      
      if (MV_OK == ret) {
        // 配置转换参数
        convert_param.nWidth = out_frame.stFrameInfo.nWidth;
        convert_param.nHeight = out_frame.stFrameInfo.nHeight;
        convert_param.pSrcData = out_frame.pBufAddr;
        convert_param.nSrcDataLen = out_frame.stFrameInfo.nFrameLen;
        convert_param.enSrcPixelType = out_frame.stFrameInfo.enPixelType;
        convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed; // 统一转换为BGR8

        // 设置图像消息参数
        image_msg.header.stamp = this->now();
        image_msg.height = out_frame.stFrameInfo.nHeight;
        image_msg.width = out_frame.stFrameInfo.nWidth;
        image_msg.step = out_frame.stFrameInfo.nWidth * 3; // BGR8
        image_msg.encoding = "bgr8";
        
        // 调整数据缓冲区大小
        size_t required_size = image_msg.step * image_msg.height;
        if (image_msg.data.size() != required_size) {
          image_msg.data.resize(required_size);
        }

        convert_param.pDstBuffer = image_msg.data.data();
        convert_param.nDstBufferSize = image_msg.data.size();

        // 转换像素格式
        int convert_ret = MV_CC_ConvertPixelType(camera_handle_, &convert_param);
        if (convert_ret == MV_OK) {
          // 发布图像
          auto camera_info_msg = camera_info_manager_->getCameraInfo();
          camera_info_msg.header = image_msg.header;
          camera_pub_.publish(image_msg, camera_info_msg);
          
          fail_count = 0;
        } else {
          RCLCPP_WARN(this->get_logger(), "Pixel conversion failed! ret[0x%x]", convert_ret);
        }

        MV_CC_FreeImageBuffer(camera_handle_, &out_frame);
      } else {
        RCLCPP_WARN(this->get_logger(), "Get image buffer failed! ret[0x%x]", ret);
        fail_count++;
        
        if (fail_count >= max_fail_count) {
          RCLCPP_ERROR(this->get_logger(), "Too many consecutive failures, disconnecting camera");
          camera_connected_ = false;
          closeCamera();
          break;
        }
      }
    }
    
    RCLCPP_INFO(this->get_logger(), "Image capture thread exiting");
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto & param : parameters) {
      if (!camera_connected_) {
        result.successful = false;
        result.reason = "Camera not connected, cannot set parameter: " + param.get_name();
        continue;
      }

      if (param.get_name() == "exposure_time") {
        bool auto_exposure = this->get_parameter("auto_exposure").as_bool();
        if (!auto_exposure) {
          int status = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", param.as_double());
          if (MV_OK == status) {
            RCLCPP_INFO(this->get_logger(), "Exposure time set to: %.1f", param.as_double());
          } else {
            result.successful = false;
            result.reason = "Failed to set exposure time, status = " + std::to_string(status);
          }
        }
      } else if (param.get_name() == "gain") {
        bool auto_gain = this->get_parameter("auto_gain").as_bool();
        if (!auto_gain) {
          int status = MV_CC_SetFloatValue(camera_handle_, "Gain", param.as_double());
          if (MV_OK == status) {
            RCLCPP_INFO(this->get_logger(), "Gain set to: %.1f", param.as_double());
          } else {
            result.successful = false;
            result.reason = "Failed to set gain, status = " + std::to_string(status);
          }
        }
      } else if (param.get_name() == "frame_rate") {
        int status = MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", param.as_double());
        if (MV_OK == status) {
          RCLCPP_INFO(this->get_logger(), "Frame rate set to: %.1f", param.as_double());
        } else {
          result.successful = false;
          result.reason = "Failed to set frame rate, status = " + std::to_string(status);
        }
      } else if (param.get_name() == "auto_exposure") {
        int status = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", param.as_bool() ? 1 : 0);
        if (MV_OK == status) {
          RCLCPP_INFO(this->get_logger(), "Auto exposure %s", param.as_bool() ? "enabled" : "disabled");
        } else {
          result.successful = false;
          result.reason = "Failed to set auto exposure, status = " + std::to_string(status);
        }
      } else if (param.get_name() == "auto_gain") {
        int status = MV_CC_SetEnumValue(camera_handle_, "GainAuto", param.as_bool() ? 1 : 0);
        if (MV_OK == status) {
          RCLCPP_INFO(this->get_logger(), "Auto gain %s", param.as_bool() ? "enabled" : "disabled");
        } else {
          result.successful = false;
          result.reason = "Failed to set auto gain, status = " + std::to_string(status);
        }
      } else {
        RCLCPP_WARN(this->get_logger(), "Unknown parameter: %s", param.get_name().c_str());
      }
    }

    return result;
  }

  // 成员变量
  void * camera_handle_ = nullptr;
  std::atomic<bool> running_{true};
  std::atomic<bool> camera_connected_{false};
  
  std::thread camera_manager_thread_;
  std::thread capture_thread_;
  
  image_transport::CameraPublisher camera_pub_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  
  OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
};

}  // namespace hik_camera

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(hik_camera::HikCameraNode)