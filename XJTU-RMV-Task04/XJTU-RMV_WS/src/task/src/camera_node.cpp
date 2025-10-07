// camera_node.cpp
// Hikvision MVS SDK (USB) -> ROS2 image publisher
// - Uses MV_CC_GetImageBuffer + MV_CC_ConvertPixelType
// - Reads parameters from ROS2 parameter server (so launch can load YAML)
// - Publishes image_transport camera publisher (image + camera_info)
//
// Build: part of a ament package; link with MvCameraControl and related libs.

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>
#include <camera_info_manager/camera_info_manager.hpp>

#include "MvCameraControl.h"

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <arpa/inet.h> // ntohl if needed for GigE (safe to include)
#include <cstring>

using namespace std::chrono_literals;

class HikCameraNode : public rclcpp::Node
{
public:
  explicit HikCameraNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
  : Node("hik_camera_node", options)
  {
    RCLCPP_INFO(get_logger(), "[HikCameraNode] start");

    // --- Declare parameters (defaults) ---
    this->declare_parameter<std::string>("camera_sn", "");
    this->declare_parameter<std::string>("camera_ip", "");
    this->declare_parameter<double>("exposure_time", 50000.0); // microseconds
    this->declare_parameter<double>("gain", 10.0);
    this->declare_parameter<double>("frame_rate", 30.0);
    this->declare_parameter<std::string>("pixel_format", "BGR8"); // BGR8 | Mono8
    this->declare_parameter<int64_t>("width", 640);
    this->declare_parameter<int64_t>("height", 480);
    this->declare_parameter<int>("reconnect_interval_sec", 2);
    this->declare_parameter<int>("max_reconnect_attempts", 0); // 0 -> infinite

    // camera_info YAML URL (optional)
    this->declare_parameter<std::string>("camera_info_url", std::string(""));

    // QoS choice
    this->declare_parameter<bool>("use_sensor_data_qos", false);

    bool use_sensor_qos = this->get_parameter("use_sensor_data_qos").as_bool();
    auto qos = use_sensor_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;

    // publisher (camera publisher expects image + camera_info)
    camera_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);

    // camera_info manager
    camera_name_ = this->get_parameter("camera_sn").as_string(); // fallback; can be overridden
    if (camera_name_.empty()) {
      camera_name_ = this->get_name();
    }
    camera_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    std::string camera_info_url = this->get_parameter("camera_info_url").as_string();
    if (!camera_info_url.empty() && camera_info_manager_->validateURL(camera_info_url)) {
      camera_info_manager_->loadCameraInfo(camera_info_url);
      camera_info_msg_ = camera_info_manager_->getCameraInfo();
      RCLCPP_INFO(get_logger(), "[HikCameraNode] loaded camera_info from %s", camera_info_url.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "[HikCameraNode] no valid camera_info_url provided or file missing");
    }

    // Register parameter callback for runtime changes
    params_cb_handle_ = this->add_on_set_parameters_callback(
      std::bind(&HikCameraNode::on_parameter_change, this, std::placeholders::_1));

    // Start capture thread (will attempt to connect and capture)
    running_.store(true);
    capture_thread_ = std::thread(&HikCameraNode::captureLoop, this);
  }

  ~HikCameraNode() override
  {
    running_.store(false);
    if (capture_thread_.joinable()) capture_thread_.join();
    disconnectCamera();
    RCLCPP_INFO(get_logger(), "[HikCameraNode] stopped");
  }

private:
  // --- members ---
  void *camera_handle_ = nullptr;
  std::atomic<bool> running_{false};
  std::thread capture_thread_;
  std::mutex camera_mutex_;

  std::unique_ptr<unsigned char[]> sdk_buffer_; // raw buffer from SDK (payload)
  size_t sdk_buffer_size_ = 0;

  image_transport::CameraPublisher camera_pub_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;
  std::string camera_name_;

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr params_cb_handle_;

  // helper: get param value safely
  template<typename T>
  T get_param_or(const std::string &name, const T &def) {
    try {
      return this->get_parameter(name).get_value<T>();
    } catch (...) {
      return def;
    }
  }

  // parameter change callback
  rcl_interfaces::msg::SetParametersResult on_parameter_change(const std::vector<rclcpp::Parameter> &params)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto &p : params) {
      RCLCPP_INFO(get_logger(), "[on_param] %s changed", p.get_name().c_str());
      // For parameters that require restarting grabbing (pixel_format / resolution), we enforce reconnect in capture loop
    }
    return result;
  }

  // connect to camera (USB/GIGE depending on enum)
  bool connectCamera()
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);

    // Enum USB devices (use MV_USB_DEVICE for USB)
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(stDeviceList));
    int nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (nRet != MV_OK) {
      RCLCPP_ERROR(get_logger(), "[connectCamera] MV_CC_EnumDevices failed: 0x%X", nRet);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[connectCamera] found %d devices", stDeviceList.nDeviceNum);
    if (stDeviceList.nDeviceNum == 0) {
      return false;
    }

    // choose device: if camera_sn provided, try match; else first device
    std::string wanted_sn = get_param_or<std::string>("camera_sn", std::string(""));
    int selected_index = 0;
    if (!wanted_sn.empty()) {
      bool found = false;
      for (unsigned int i = 0; i < stDeviceList.nDeviceNum; ++i) {
        MV_CC_DEVICE_INFO *info = stDeviceList.pDeviceInfo[i];
        // read serial number using SDK struct MVCC_DEVICE_SF_INFO? prefer MVCC_STRINGVALUE via MV_CC_GetStringValue
        MVCC_STRINGVALUE stSn;
        memset(&stSn, 0, sizeof(stSn));
        // MV_CC_GetStringValue has overloads for handle options, but SDK provides ways to read device info.
        // Use MV_CC_GetStringValue with the device info pointer is not available on all SDKs; if not available, fallback to pDeviceInfo data access.
        int sret = MV_CC_GetStringValue(info, "DeviceSerialNumber", &stSn);
        if (sret == MV_OK) {
          std::string cur_sn = stSn.chCurValue;
          if (cur_sn == wanted_sn) { selected_index = i; found = true; break; }
        } else {
          // Fallback: try to read SpecialInfo if present (some SDK builds)
          // Many USB device info structs include SerialNumber in SpecialInfo, but name differs. We skip fallback here.
        }
      }
      if (!found) {
        RCLCPP_WARN(get_logger(), "[connectCamera] SN '%s' not found, using first device", wanted_sn.c_str());
      }
    }

    // create handle
    nRet = MV_CC_CreateHandle(&camera_handle_, stDeviceList.pDeviceInfo[selected_index]);
    if (nRet != MV_OK) {
      RCLCPP_ERROR(get_logger(), "[connectCamera] MV_CC_CreateHandle failed: 0x%X", nRet);
      camera_handle_ = nullptr;
      return false;
    }

    // open device
    nRet = MV_CC_OpenDevice(camera_handle_);
    if (nRet != MV_OK) {
      RCLCPP_ERROR(get_logger(), "[connectCamera] MV_CC_OpenDevice failed: 0x%X", nRet);
      MV_CC_DestroyHandle(&camera_handle_);
      camera_handle_ = nullptr;
      return false;
    }

    // set trigger off (free-run)
    MV_CC_SetEnumValue(camera_handle_, "TriggerMode", MV_TRIGGER_MODE_OFF);

    // apply parameters from server (resolution / exposure / gain / frame_rate / pixel_format)
    int64_t width = get_param_or<int64_t>("width", 640);
    int64_t height = get_param_or<int64_t>("height", 480);
    double exposure = get_param_or<double>("exposure_time", 50000.0);
    double gain = get_param_or<double>("gain", 10.0);
    double frame_rate = get_param_or<double>("frame_rate", 30.0);
    std::string pixel_format = get_param_or<std::string>("pixel_format", std::string("BGR8"));

    // set resolution (some cameras require stop/start to set resolution; here attempt to set before grabbing)
    MV_CC_SetIntValue(camera_handle_, "Width", width);
    MV_CC_SetIntValue(camera_handle_, "Height", height);

    // set exposure/gain/fps
    MV_CC_SetFloatValue(camera_handle_, "ExposureTime", exposure);
    MV_CC_SetFloatValue(camera_handle_, "Gain", gain);
    MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", frame_rate);

    // set PixelFormat on device if supported (we will still convert via SDK to bgr8)
    if (pixel_format == "Mono8") {
      MV_CC_SetEnumValue(camera_handle_, "PixelFormat", PixelType_Gvsp_Mono8);
    } else if (pixel_format == "BGR8") {
      MV_CC_SetEnumValue(camera_handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
    } else if (pixel_format == "RGB8") {
      // many Hik cameras use BGR ordering; set to BGR on device and convert later if needed
      MV_CC_SetEnumValue(camera_handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
    }

    // get payload size from SDK and allocate sdk_buffer_
    MVCC_INTVALUE_EX stPayload;
    memset(&stPayload, 0, sizeof(stPayload));
    nRet = MV_CC_GetIntValueEx(camera_handle_, "PayloadSize", &stPayload);
    if (nRet != MV_OK) {
      RCLCPP_ERROR(get_logger(), "[connectCamera] MV_CC_GetIntValueEx(PayloadSize) failed: 0x%X", nRet);
      disconnectCamera();
      return false;
    }
    sdk_buffer_size_ = static_cast<size_t>(stPayload.nCurValue);
    sdk_buffer_.reset(new unsigned char[sdk_buffer_size_]);

    // start grabbing
    nRet = MV_CC_StartGrabbing(camera_handle_);
    if (nRet != MV_OK) {
      RCLCPP_ERROR(get_logger(), "[connectCamera] MV_CC_StartGrabbing failed: 0x%X", nRet);
      disconnectCamera();
      return false;
    }

    // adjust publish rate: cap frame_rate to reasonable max to avoid timeouts
    double cap_rate = std::min(frame_rate, 30.0);
    publish_period_ms_ = static_cast<int>(1000.0 / cap_rate);

    RCLCPP_INFO(get_logger(), "[connectCamera] camera opened and grabbing started (payload=%zu bytes).", sdk_buffer_size_);
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
    }
    sdk_buffer_.reset();
    sdk_buffer_size_ = 0;
  }

  void captureLoop()
  {
    // Wait a moment at startup to let USB / kernel enumerate device
    std::this_thread::sleep_for(2000ms);

    int reconnect_interval = get_param_or<int>("reconnect_interval_sec", 2);
    int max_retries = get_param_or<int>("max_reconnect_attempts", 0);

    int attempt = 0;
    while (running_.load() && rclcpp::ok()) {
      {
        std::lock_guard<std::mutex> lock(camera_mutex_);
        if (!camera_handle_) {
          // try connect
          if (!connectCamera()) {
            ++attempt;
            if (max_retries > 0 && attempt >= max_retries) {
              RCLCPP_ERROR(get_logger(), "[captureLoop] max reconnect attempts reached, exiting capture loop");
              return;
            }
            std::this_thread::sleep_for(std::chrono::seconds(reconnect_interval));
            continue;
          }
          attempt = 0;
        }
      } // unlock

      // Grab one frame using recommended API
      MV_FRAME_OUT stOutFrame;
      memset(&stOutFrame, 0, sizeof(stOutFrame));
      int nRet = MV_CC_GetImageBuffer(camera_handle_, &stOutFrame, 1000);
      if (nRet != MV_OK) {
        if (nRet == static_cast<int>(0x80000007)) { // timeout
          RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000, "[captureLoop] MV_CC_GetImageBuffer timeout (0x%X).", nRet);
          std::this_thread::sleep_for(5ms);
          continue;
        } else {
          RCLCPP_ERROR(get_logger(), "[captureLoop] MV_CC_GetImageBuffer failed: 0x%X", nRet);
          // attempt reconnect
          disconnectCamera();
          std::this_thread::sleep_for(std::chrono::seconds(reconnect_interval));
          continue;
        }
      }

      // Got raw frame in stOutFrame.pBufAddr
      // Convert to BGR8 (or Mono8) using SDK convert API
      MV_CC_PIXEL_CONVERT_PARAM convertParam;
      memset(&convertParam, 0, sizeof(convertParam));
      convertParam.nWidth = stOutFrame.stFrameInfo.nWidth;
      convertParam.nHeight = stOutFrame.stFrameInfo.nHeight;
      convertParam.pSrcData = stOutFrame.pBufAddr;
      convertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen;
      convertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType;

      // target encoding: bgr8 if these are color, mono8 if mono
      bool is_mono = (stOutFrame.stFrameInfo.enPixelType == PixelType_Gvsp_Mono8);
      convertParam.enDstPixelType = is_mono ? PixelType_Gvsp_Mono8 : PixelType_Gvsp_BGR8_Packed;

      // allocate a conversion buffer sized width*height*channels (safe upper bound)
      size_t channels = is_mono ? 1 : 3;
      size_t expected_size = static_cast<size_t>(stOutFrame.stFrameInfo.nWidth) * stOutFrame.stFrameInfo.nHeight * channels;

      // allocate temporary buffer for converted image (use unique_ptr)
      std::unique_ptr<unsigned char[]> convert_buffer(new unsigned char[expected_size]);
      convertParam.pDstBuffer = convert_buffer.get();
      convertParam.nDstBufferSize = static_cast<unsigned int>(expected_size);

      int conv_ret = MV_CC_ConvertPixelType(camera_handle_, &convertParam);
      if (conv_ret != MV_OK) {
        RCLCPP_ERROR(get_logger(), "[captureLoop] MV_CC_ConvertPixelType failed: 0x%X", conv_ret);
        MV_CC_FreeImageBuffer(camera_handle_, &stOutFrame);
        std::this_thread::sleep_for(5ms);
        continue;
      }

      // Prepare ROS Image message
      sensor_msgs::msg::Image img;
      img.header.stamp = this->now();
      img.header.frame_id = "camera_optical_frame";
      img.height = stOutFrame.stFrameInfo.nHeight;
      img.width = stOutFrame.stFrameInfo.nWidth;
      img.is_bigendian = 0;
      img.step = static_cast<sensor_msgs::msg::Image::_step_type>(stOutFrame.stFrameInfo.nWidth * channels);
      img.encoding = is_mono ? "mono8" : "bgr8";

      // Copy converted buffer to msg
      img.data.resize(expected_size);
      std::memcpy(img.data.data(), convert_buffer.get(), expected_size);

      // camera_info timestamp/header
      camera_info_msg_.header = img.header;

      // publish (image_transport camera publisher)
      camera_pub_.publish(img, camera_info_msg_);

      // free SDK buffer
      MV_CC_FreeImageBuffer(camera_handle_, &stOutFrame);

      // throttle loop to configured publish period to avoid busy spin
      std::this_thread::sleep_for(std::chrono::milliseconds(publish_period_ms_));
    } // while
  }

  int publish_period_ms_ = 33; // default ~30fps
};

// Register as component if desired, here provide main for standalone node
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HikCameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}