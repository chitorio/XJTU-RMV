#include "MvCameraControl.h"

// ROS
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <thread>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace hik_camera
{

class HikCameraNode : public rclcpp::Node
{
public:
  explicit HikCameraNode(const rclcpp::NodeOptions & options) : Node("hik_camera", options)
  {
    RCLCPP_INFO(this->get_logger(), "Starting HikCameraNode...");

    // Declare parameters
    camera_sn_ = this->declare_parameter<std::string>("camera_sn", "");
    camera_ip_ = this->declare_parameter<std::string>("camera_ip", "");
    use_sensor_data_qos_ = this->declare_parameter<bool>("use_sensor_data_qos", true);

    // QoS
    auto qos = use_sensor_data_qos_ ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    camera_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);

    // Try to connect to camera
    if (!connectCamera()) {
      RCLCPP_FATAL(this->get_logger(), "Failed to connect to camera. Shutting down.");
      rclcpp::shutdown();
      return;
    }

    // Load camera info
    camera_name_ = this->declare_parameter("camera_name", "narrow_stereo");
    camera_info_manager_ =
      std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    auto camera_info_url = this->declare_parameter("camera_info_url", "package://hik_camera/config/camera_info.yaml");
    if (camera_info_manager_->validateURL(camera_info_url)) {
      camera_info_manager_->loadCameraInfo(camera_info_url);
      camera_info_msg_ = camera_info_manager_->getCameraInfo();
    } else {
      RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s", camera_info_url.c_str());
    }

    // Declare dynamic parameters for camera settings
    declareCameraParameters();

    // Start capture thread
    capture_thread_ = std::thread([this]() { captureLoop(); });

    // Set parameters callback
    params_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&HikCameraNode::parametersCallback, this, std::placeholders::_1));
  }

  ~HikCameraNode() override
  {
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    disconnectCamera();
    RCLCPP_INFO(this->get_logger(), "HikCameraNode destroyed.");
  }

private:
  // ----------------- Camera Connection -----------------
  bool connectCamera()
  {
    MV_CC_DEVICE_INFO_LIST device_list;
    nRet_ = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
    RCLCPP_INFO(this->get_logger(), "Found camera count = %d", device_list.nDeviceNum);

    while (device_list.nDeviceNum == 0 && rclcpp::ok()) {
        RCLCPP_WARN(this->get_logger(), "No camera found, retrying in 1 second...");
        std::this_thread::sleep_for(std::chrono::seconds(1));
        nRet_ = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
    }

    if (device_list.nDeviceNum == 0) {
        RCLCPP_ERROR(this->get_logger(), "No cameras available!");
        return false;
    }

    // 默认选第一个设备
    MV_CC_DEVICE_INFO* selected_dev = device_list.pDeviceInfo[0];

    // 如果指定了 SN/IP，则遍历寻找
    if (!camera_sn_.empty() || !camera_ip_.empty()) {
        selected_dev = nullptr;
        for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
            MV_CC_DEVICE_INFO* dev = device_list.pDeviceInfo[i];
            if ((camera_sn_.size() &&
                 camera_sn_ == reinterpret_cast<char*>(dev->SpecialInfo.stGigEInfo.chSerialNumber)) ||
                (camera_ip_.size() &&
                 camera_ip_ == std::to_string(dev->SpecialInfo.stGigEInfo.nCurrentIp))) {
                selected_dev = dev;
                break;
            }
        }
        if (!selected_dev) {
            RCLCPP_ERROR(this->get_logger(), "Specified camera SN/IP not found!");
            return false;
        }
    }

    nRet_ = MV_CC_CreateHandle(&camera_handle_, selected_dev);
    if (nRet_ != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Failed to create camera handle!");
        return false;
    }

    nRet_ = MV_CC_OpenDevice(camera_handle_);
    if (nRet_ != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open camera device!");
        return false;
    }

    nRet_ = MV_CC_GetImageInfo(camera_handle_, &img_info_);
    if (nRet_ != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get image info!");
        return false;
    }

    image_msg_.data.resize(img_info_.nWidthMax * img_info_.nHeightMax * 3);

    convert_param_.nWidth = img_info_.nWidthValue;
    convert_param_.nHeight = img_info_.nHeightValue;
    convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
    convert_param_.pDstBuffer = image_msg_.data.data();
    convert_param_.nDstBufferSize = image_msg_.data.size();

    nRet_ = MV_CC_StartGrabbing(camera_handle_);
    if (nRet_ != MV_OK) {
        RCLCPP_ERROR(this->get_logger(), "Failed to start grabbing!");
        return false;
    }

    return true;
  }

  void disconnectCamera()
  {
    if (camera_handle_) {
      MV_CC_StopGrabbing(camera_handle_);
      MV_CC_CloseDevice(camera_handle_);
      MV_CC_DestroyHandle(&camera_handle_);
      camera_handle_ = nullptr;
    }
  }

  // ----------------- Camera Parameters -----------------
  void declareCameraParameters()
  {
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    MVCC_FLOATVALUE f_value;

    // Exposure
    MV_CC_GetFloatValue(camera_handle_, "ExposureTime", &f_value);
    param_desc.description = "Exposure time in microseconds";
    param_desc.integer_range.resize(1);
    param_desc.integer_range[0].from_value = f_value.fMin;
    param_desc.integer_range[0].to_value = f_value.fMax;
    param_desc.integer_range[0].step = 1;
    double exposure_time = this->declare_parameter("exposure_time", f_value.fCurValue, param_desc);
    MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure_time);

    // Gain
    MV_CC_GetFloatValue(camera_handle_, "Gain", &f_value);
    param_desc.description = "Camera gain";
    param_desc.integer_range[0].from_value = f_value.fMin;
    param_desc.integer_range[0].to_value = f_value.fMax;
    double gain = this->declare_parameter("gain", f_value.fCurValue, param_desc);
    MV_CC_SetFloatValue(camera_handle_, "Gain", gain);

    // Frame Rate
    MV_CC_GetFloatValue(camera_handle_, "AcquisitionFrameRate", &f_value);
    param_desc.description = "Frame rate";
    param_desc.integer_range[0].from_value = f_value.fMin;
    param_desc.integer_range[0].to_value = f_value.fMax;
    double frame_rate = this->declare_parameter("frame_rate", f_value.fCurValue, param_desc);
    MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", frame_rate);

    // Pixel Format
    std::string default_pixel = "RGB8";
    pixel_format_ = this->declare_parameter("pixel_format", default_pixel);
    if (pixel_format_ == "Mono8") convert_param_.enDstPixelType = PixelType_Gvsp_Mono8;
    else convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
      const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (auto & param : parameters) {
      if (param.get_name() == "exposure_time") {
        nRet_ = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", param.as_double());
      } else if (param.get_name() == "gain") {
        nRet_ = MV_CC_SetFloatValue(camera_handle_, "Gain", param.as_double());
      } else if (param.get_name() == "frame_rate") {
        nRet_ = MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", param.as_double());
      } else if (param.get_name() == "pixel_format") {
        pixel_format_ = param.as_string();
        if (pixel_format_ == "Mono8") convert_param_.enDstPixelType = PixelType_Gvsp_Mono8;
        else convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
      } else {
        result.successful = false;
        result.reason = "Unknown parameter: " + param.get_name();
      }
      if (nRet_ != MV_OK) {
        result.successful = false;
        result.reason += " Failed to apply parameter!";
      }
    }
    return result;
  }

  // ----------------- Capture Loop -----------------
  void captureLoop()
  {
    MV_FRAME_OUT frame;
    fail_count_ = 0;

    while (rclcpp::ok()) {
      if (!camera_handle_) {
        if (!connectCamera()) {
          std::this_thread::sleep_for(std::chrono::seconds(2));
          continue;
        }
      }

      nRet_ = MV_CC_GetImageBuffer(camera_handle_, &frame, 1000);
      if (nRet_ == MV_OK) {
        convert_param_.pSrcData = frame.pBufAddr;
        convert_param_.nSrcDataLen = frame.stFrameInfo.nFrameLen;
        MV_CC_ConvertPixelType(camera_handle_, &convert_param_);

        image_msg_.header.stamp = now();
        image_msg_.header.frame_id = "camera_optical_frame";
        image_msg_.height = frame.stFrameInfo.nHeight;
        image_msg_.width = frame.stFrameInfo.nWidth;
        image_msg_.step = frame.stFrameInfo.nWidth * 3;
        image_msg_.data.resize(image_msg_.width * image_msg_.height * 3);

        camera_info_msg_.header = image_msg_.header;
        camera_pub_.publish(image_msg_, camera_info_msg_);

        MV_CC_FreeImageBuffer(camera_handle_, &frame);
        fail_count_ = 0;
      } else {
        RCLCPP_WARN(this->get_logger(), "Get buffer failed! nRet=[%x]", nRet_);
        fail_count_++;
        if (fail_count_ > 5) {
          RCLCPP_WARN(this->get_logger(), "Camera lost, attempting reconnect...");
          disconnectCamera();
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }
      }
    }
  }

private:
  void* camera_handle_ = nullptr;
  int nRet_ = MV_OK;
  MV_IMAGE_BASIC_INFO img_info_;
  MV_CC_PIXEL_CONVERT_PARAM convert_param_;
  sensor_msgs::msg::Image image_msg_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  image_transport::CameraPublisher camera_pub_;
  std::thread capture_thread_;
  int fail_count_ = 0;

  std::string camera_name_;
  std::string camera_sn_;
  std::string camera_ip_;
  std::string pixel_format_;
  bool use_sensor_data_qos_;
  OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
};

}  // namespace hik_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hik_camera::HikCameraNode)
