// camera_node.cpp
#include "MvCameraControl.h"

// ROS
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <thread>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iomanip>

namespace hik_camera
{

class HikCameraNode : public rclcpp::Node
{
public:
  explicit HikCameraNode(const rclcpp::NodeOptions & options) : Node("camera_node", options)
  {
    RCLCPP_INFO(this->get_logger(), "[HikCameraNode] start");

    // Basic connection selection params (declare before trying to read CLI)
    camera_sn_ = this->declare_parameter<std::string>("camera_sn", "");
    camera_ip_ = this->declare_parameter<std::string>("camera_ip", "");
    use_sensor_data_qos_ = this->declare_parameter<bool>("use_sensor_data_qos", true);

    reconnect_interval_sec_ =
      this->declare_parameter<int>("reconnect_interval_sec", 2);
    max_reconnect_attempts_ =
      this->declare_parameter<int>("max_reconnect_attempts", 0); // 0 => infinite

    // QoS and publisher
    auto qos = use_sensor_data_qos_ ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    camera_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);

    // Try connecting. We'll try a few times here before giving control to capture loop.
    if (!connectCamera()) {
      RCLCPP_WARN(this->get_logger(), "[HikCameraNode] initial connect failed. Capture loop will attempt reconnects.");
      // do not shutdown here — captureLoop includes reconnect attempts
    }

    // CameraInfo manager (optional file)
    camera_name_ = this->declare_parameter<std::string>("camera_name", "narrow_stereo");
    camera_info_manager_ =
      std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    auto camera_info_url = this->declare_parameter<std::string>(
      "camera_info_url", "package://task/config/camera_info.yaml");
    if (camera_info_manager_->validateURL(camera_info_url)) {
      camera_info_manager_->loadCameraInfo(camera_info_url);
      camera_info_msg_ = camera_info_manager_->getCameraInfo();
    } else {
      RCLCPP_INFO(this->get_logger(), "[HikCameraNode] camera_info URL invalid or not provided");
    }

    // Declare and apply camera parameters that require an open handle
    // Note: if camera not open yet, declareCameraParameters() will skip SDK calls.
    declareCameraParameters();

    // Parameter change callback
    params_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&HikCameraNode::parametersCallback, this, std::placeholders::_1));

    // Start capture thread
    running_.store(true);
    capture_thread_ = std::thread(&HikCameraNode::captureLoop, this);
  }

  ~HikCameraNode() override
  {
    RCLCPP_INFO(this->get_logger(), "[HikCameraNode] shutting down...");
    running_.store(false);
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    disconnectCamera();
    RCLCPP_INFO(this->get_logger(), "[HikCameraNode] stopped.");
  }

private:
  // ---------------- helpers ----------------
  static std::string ipToString(uint32_t ip)
  {
    // SDK might store IP in host or network order; this produces a dotted quad using big-endian order.
    std::ostringstream ss;
    ss << ((ip >> 24) & 0xFF) << "." << ((ip >> 16) & 0xFF) << "." << ((ip >> 8) & 0xFF) << "." << (ip & 0xFF);
    return ss.str();
  }

  // ---------------- Camera connection ----------------
  bool connectCamera()
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);

    MV_CC_DEVICE_INFO_LIST device_list;
    nRet_ = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
    if (nRet_ != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "[connectCamera] MV_CC_EnumDevices failed, nRet=0x%X", nRet_);
      return false;
    }
    RCLCPP_INFO(this->get_logger(), "[connectCamera] found %d devices", device_list.nDeviceNum);

    if (device_list.nDeviceNum == 0) {
      return false;
    }

    // Choose default device (first) or match by SN/IP if provided
    MV_CC_DEVICE_INFO* selected_dev = device_list.pDeviceInfo[0];

    if (!camera_sn_.empty() || !camera_ip_.empty()) {
      selected_dev = nullptr;
      for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
        MV_CC_DEVICE_INFO* dev = device_list.pDeviceInfo[i];
        bool match = false;
        // Try SN (GigE has serial in stGigEInfo.chSerialNumber)
        if (!camera_sn_.empty()) {
          if (std::strncmp(reinterpret_cast<char*>(dev->SpecialInfo.stGigEInfo.chSerialNumber),
                           camera_sn_.c_str(), camera_sn_.size()) == 0) {
            match = true;
          }
        }
        // Try IP matching
        if (!match && !camera_ip_.empty()) {
          uint32_t ip_val = dev->SpecialInfo.stGigEInfo.nCurrentIp;
          if (!camera_ip_.empty() && camera_ip_ == ipToString(ip_val)) {
            match = true;
          }
        }
        if (match) {
          selected_dev = dev;
          break;
        }
      }
      if (!selected_dev) {
        RCLCPP_WARN(this->get_logger(), "[connectCamera] specified SN/IP not found");
        return false;
      }
    }

    // Create & open handle
    nRet_ = MV_CC_CreateHandle(&camera_handle_, selected_dev);
    if (nRet_ != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "[connectCamera] MV_CC_CreateHandle failed, nRet=0x%X", nRet_);
      camera_handle_ = nullptr;
      return false;
    }

    nRet_ = MV_CC_OpenDevice(camera_handle_);
    if (nRet_ != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "[connectCamera] MV_CC_OpenDevice failed, nRet=0x%X", nRet_);
      MV_CC_DestroyHandle(&camera_handle_);
      camera_handle_ = nullptr;
      return false;
    }

    // Get image info
    nRet_ = MV_CC_GetImageInfo(camera_handle_, &img_info_);
    if (nRet_ != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "[connectCamera] MV_CC_GetImageInfo failed, nRet=0x%X", nRet_);
      disconnectCamera();
      return false;
    }

    // allocate stable frame buffer (max size)
    size_t max_channels = 3; // we'll assume RGB max; Mono8 will use 1 channel later
    frame_buffer_.resize(static_cast<size_t>(img_info_.nWidthMax) * img_info_.nHeightMax * max_channels);

    // Setup convert param (point to frame_buffer_)
    convert_param_.nWidth = img_info_.nWidthValue;
    convert_param_.nHeight = img_info_.nHeightValue;
    // enDstPixelType set by declareCameraParameters / parameter callback; default to RGB
    if (pixel_format_.empty()) {
      convert_param_.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
      pixel_format_ = "RGB8";
    } else {
      convert_param_.enDstPixelType = (pixel_format_ == "Mono8") ? PixelType_Gvsp_Mono8 : PixelType_Gvsp_RGB8_Packed;
    }
    convert_param_.pDstBuffer = frame_buffer_.data();
    convert_param_.nDstBufferSize = frame_buffer_.size();

    // Start grabbing
    nRet_ = MV_CC_StartGrabbing(camera_handle_);
    if (nRet_ != MV_OK) {
      RCLCPP_ERROR(this->get_logger(), "[connectCamera] MV_CC_StartGrabbing failed, nRet=0x%X", nRet_);
      disconnectCamera();
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "[connectCamera] camera opened and grabbing started");
    return true;
  }

  void disconnectCamera()
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    if (camera_handle_) {
      MV_CC_StopGrabbing(camera_handle_);
      MV_CC_CloseDevice(camera_handle_);
      MV_CC_DestroyHandle(&camera_handle_);
      camera_handle_ = nullptr;
      RCLCPP_INFO(this->get_logger(), "[disconnectCamera] camera disconnected");
    }
  }

  // ---------------- Camera parameters ----------------
  void declareCameraParameters()
  {
    // This function declares parameters. If camera_handle_ exists, we also query SDK defaults/limits.
    try {
      rcl_interfaces::msg::ParameterDescriptor desc;
      MVCC_FLOATVALUE fvalue;

      if (camera_handle_) {
        // Exposure
        if (MV_CC_GetFloatValue(camera_handle_, "ExposureTime", &fvalue) == MV_OK) {
          desc.description = "Exposure time (microseconds)";
          desc.integer_range.resize(1);
          desc.integer_range[0].from_value = fvalue.fMin;
          desc.integer_range[0].to_value = fvalue.fMax;
          desc.integer_range[0].step = 1;
          double exposure_default = static_cast<double>(fvalue.fCurValue);
          double exposure = declare_parameter("exposure_time", exposure_default, desc);
          MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure);
        } else {
          declare_parameter("exposure_time", 5000.0);
        }

        // Gain
        if (MV_CC_GetFloatValue(camera_handle_, "Gain", &fvalue) == MV_OK) {
          desc.description = "Camera gain";
          desc.integer_range.resize(1);
          desc.integer_range[0].from_value = fvalue.fMin;
          desc.integer_range[0].to_value = fvalue.fMax;
          desc.integer_range[0].step = 1;
          double gain_default = static_cast<double>(fvalue.fCurValue);
          double gain = declare_parameter("gain", gain_default, desc);
          MV_CC_SetFloatValue(camera_handle_, "Gain", gain);
        } else {
          declare_parameter("gain", 0.0);
        }

        // Frame rate
        if (MV_CC_GetFloatValue(camera_handle_, "AcquisitionFrameRate", &fvalue) == MV_OK) {
          desc.description = "Acquisition frame rate";
          desc.integer_range.resize(1);
          desc.integer_range[0].from_value = fvalue.fMin;
          desc.integer_range[0].to_value = fvalue.fMax;
          desc.integer_range[0].step = 1;
          double fr_default = static_cast<double>(fvalue.fCurValue);
          double fr = declare_parameter("frame_rate", fr_default, desc);
          MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", fr);
        } else {
          declare_parameter("frame_rate", 30.0);
        }

        // Pixel format (enum handled locally)
        std::string pf_default = "RGB8";
        pixel_format_ = declare_parameter("pixel_format", pf_default);
        {
          std::lock_guard<std::mutex> lock(camera_mutex_);
          convert_param_.enDstPixelType = (pixel_format_ == "Mono8") ? PixelType_Gvsp_Mono8 : PixelType_Gvsp_RGB8_Packed;
        }
      } else {
        // Camera not open yet; declare default params so users can set them before connect
        declare_parameter("exposure_time", 5000.0);
        declare_parameter("gain", 0.0);
        declare_parameter("frame_rate", 30.0);
        pixel_format_ = declare_parameter("pixel_format", std::string("RGB8"));
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "[declareCameraParameters] exception: %s", e.what());
    }
  }

  rcl_interfaces::msg::SetParametersResult parametersCallback(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    // Use local status; do not rely on nRet_ which is shared.
    for (const auto & p : parameters) {
      if (p.get_name() == "exposure_time") {
        double v = p.as_double();
        int status = MV_OK;
        {
          std::lock_guard<std::mutex> lock(camera_mutex_);
          if (camera_handle_) status = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", v);
        }
        if (status != MV_OK) {
          result.successful = false;
          result.reason = "Failed to set exposure_time";
          return result;
        }
      } else if (p.get_name() == "gain") {
        double v = p.as_double();
        int status = MV_OK;
        {
          std::lock_guard<std::mutex> lock(camera_mutex_);
          if (camera_handle_) status = MV_CC_SetFloatValue(camera_handle_, "Gain", v);
        }
        if (status != MV_OK) {
          result.successful = false;
          result.reason = "Failed to set gain";
          return result;
        }
      } else if (p.get_name() == "frame_rate") {
        double v = p.as_double();
        int status = MV_OK;
        {
          std::lock_guard<std::mutex> lock(camera_mutex_);
          if (camera_handle_) status = MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", v);
        }
        if (status != MV_OK) {
          result.successful = false;
          result.reason = "Failed to set frame_rate";
          return result;
        }
      } else if (p.get_name() == "pixel_format") {
        std::string v = p.as_string();
        {
          std::lock_guard<std::mutex> lock(camera_mutex_);
          pixel_format_ = v;
          convert_param_.enDstPixelType = (pixel_format_ == "Mono8") ? PixelType_Gvsp_Mono8 : PixelType_Gvsp_RGB8_Packed;
        }
      } else if (p.get_name() == "camera_sn" || p.get_name() == "camera_ip") {
        // allow change of target camera: update parameter only; reconnect handled in capture loop if necessary
        if (p.get_name() == "camera_sn") camera_sn_ = p.as_string();
        if (p.get_name() == "camera_ip") camera_ip_ = p.as_string();
        RCLCPP_INFO(this->get_logger(), "[parametersCallback] will attempt reconnect to new camera on next loop");
      } else {
        // ignore other params
      }
    }

    return result;
  }

  // ---------------- Capture Loop ----------------
  void captureLoop()
  {
    MV_FRAME_OUT frame;
    int local_fail_count = 0;
    size_t channels = (convert_param_.enDstPixelType == PixelType_Gvsp_Mono8) ? 1 : 3;

    while (rclcpp::ok() && running_.load()) {
      // Ensure camera is connected
      if (!camera_handle_) {
        bool connected = false;
        int attempts = 0;
        while (rclcpp::ok() && running_.load()) {
          if (connectCamera()) {
            connected = true;
            break;
          }
          ++attempts;
          if (max_reconnect_attempts_ > 0 && attempts >= max_reconnect_attempts_) {
            RCLCPP_ERROR(this->get_logger(), "[captureLoop] max reconnect attempts reached");
            break;
          }
          std::this_thread::sleep_for(std::chrono::seconds(reconnect_interval_sec_));
        }
        if (!connected) {
          // Sleep a bit before next outer loop iteration
          std::this_thread::sleep_for(std::chrono::seconds(reconnect_interval_sec_));
          continue;
        }
      }

      // Try to get frame
      {
        std::lock_guard<std::mutex> lock(camera_mutex_);
        nRet_ = MV_CC_GetImageBuffer(camera_handle_, &frame, 1000);
      }
      if (nRet_ == MV_OK) {
        // Convert into frame_buffer_ (protected by mutex while calling SDK)
        {
          std::lock_guard<std::mutex> lock(camera_mutex_);
          convert_param_.pSrcData = frame.pBufAddr;
          convert_param_.nSrcDataLen = frame.stFrameInfo.nFrameLen;
          // ensure convert_param_.nWidth/nHeight reflect current frame if needed
          convert_param_.nWidth = frame.stFrameInfo.nWidth;
          convert_param_.nHeight = frame.stFrameInfo.nHeight;
          MV_CC_ConvertPixelType(camera_handle_, &convert_param_);
        }

        // Determine channels and actual size
        channels = (convert_param_.enDstPixelType == PixelType_Gvsp_Mono8) ? 1 : 3;
        size_t actual_size = static_cast<size_t>(frame.stFrameInfo.nWidth) * frame.stFrameInfo.nHeight * channels;

        // Prepare and publish message
        sensor_msgs::msg::Image img;
        img.header.stamp = this->now();
        img.header.frame_id = "camera_optical_frame";
        img.height = frame.stFrameInfo.nHeight;
        img.width = frame.stFrameInfo.nWidth;
        img.step = frame.stFrameInfo.nWidth * channels;
        img.is_bigendian = 0;
        img.encoding = (channels == 1) ? "mono8" : "rgb8";

        {
          // copy only the used part of frame_buffer_
          std::lock_guard<std::mutex> lock(camera_mutex_);
          img.data.assign(frame_buffer_.begin(), frame_buffer_.begin() + actual_size);
        }

        // camera_info timestamp
        camera_info_msg_.header = img.header;

        camera_pub_.publish(img, camera_info_msg_);

        // free SDK buffer
        MV_CC_FreeImageBuffer(camera_handle_, &frame);
        local_fail_count = 0;
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "[captureLoop] MV_CC_GetImageBuffer failed: 0x%X", nRet_);
        ++local_fail_count;
        if (local_fail_count > 5) {
          RCLCPP_WARN(this->get_logger(), "[captureLoop] too many consecutive get failures, resetting connection");
          disconnectCamera();
          local_fail_count = 0;
          std::this_thread::sleep_for(std::chrono::seconds(reconnect_interval_sec_));
        }
      }
    } // end while
  }

private:
  // --- members ---
  void* camera_handle_ = nullptr;
  int nRet_ = MV_OK;
  MV_IMAGE_BASIC_INFO img_info_;
  MV_CC_PIXEL_CONVERT_PARAM convert_param_{};
  std::vector<uint8_t> frame_buffer_;
  std::mutex camera_mutex_;
  std::atomic<bool> running_{false};

  // ROS
  image_transport::CameraPublisher camera_pub_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;

  std::thread capture_thread_;
  int fail_count_ = 0;

  // params
  std::string camera_name_;
  std::string camera_sn_;
  std::string camera_ip_;
  std::string pixel_format_;
  bool use_sensor_data_qos_ = true;
  int reconnect_interval_sec_ = 2;
  int max_reconnect_attempts_ = 0; // 0 = infinite

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_callback_handle_;
};

}  // namespace hik_camera

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hik_camera::HikCameraNode)
