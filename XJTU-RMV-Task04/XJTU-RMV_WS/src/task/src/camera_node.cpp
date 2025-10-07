#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include "MvCameraControl.h"
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <functional>

class HikCameraNode : public rclcpp::Node
{
public:
    HikCameraNode() : Node("camera_node")
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Hikvision Camera Node...");

        // 声明参数
        declare_parameters();
        
        // 初始化ROS接口
        initialize_ros_interfaces();
        
        // 初始化相机信息管理器
        initialize_camera_info_manager();
        
        // 连接相机
        if (!connect_camera()) {
            throw std::runtime_error("Failed to connect to camera");
        }
        
        // 配置相机参数
        if (!configure_camera_parameters()) {
            throw std::runtime_error("Failed to configure camera parameters");
        }
        
        // 启动采集
        if (!start_grabbing()) {
            throw std::runtime_error("Failed to start image grabbing");
        }
        
        // 创建定时器用于图像采集
        setup_timer();
        
        RCLCPP_INFO(this->get_logger(), "Hikvision Camera Node initialized successfully");
    }

    ~HikCameraNode()
    {
        cleanup();
    }

private:
    // 成员变量
    void* camera_handle_ = nullptr;
    bool is_connected_ = false;
    bool is_grabbing_ = false;
    
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    image_transport::Publisher image_publisher_;
    image_transport::CameraPublisher camera_publisher_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
    
    rclcpp::TimerBase::SharedPtr grab_timer_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;
    
    std::unique_ptr<unsigned char[]> image_buffer_;
    unsigned int buffer_size_ = 0;
    
    // 参数缓存
    struct CameraParams {
        double exposure_time = 8000.0;
        double gain = 5.0;
        double frame_rate = 30.0;
        std::string pixel_format = "bgr8";
        int64_t width = 640;
        int64_t height = 480;
        std::string camera_name = "camera";
        std::string camera_frame_id = "camera_optical_frame";
        std::string camera_info_url = "";
    } current_params_;

    void declare_parameters()
    {
        // 相机基本参数
        this->declare_parameter("exposure_time", current_params_.exposure_time);
        this->declare_parameter("gain", current_params_.gain);
        this->declare_parameter("frame_rate", current_params_.frame_rate);
        this->declare_parameter("pixel_format", current_params_.pixel_format);
        this->declare_parameter("width", current_params_.width);
        this->declare_parameter("height", current_params_.height);
        
        // ROS相关参数
        this->declare_parameter("camera_name", current_params_.camera_name);
        this->declare_parameter("camera_frame_id", current_params_.camera_frame_id);
        this->declare_parameter("camera_info_url", current_params_.camera_info_url);
        
        // 从参数服务器获取最新值
        this->get_parameter("camera_name", current_params_.camera_name);
        this->get_parameter("camera_frame_id", current_params_.camera_frame_id);
        this->get_parameter("camera_info_url", current_params_.camera_info_url);
    }

    void initialize_ros_interfaces()
    {
        // 创建图像传输
        image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
        
        // 创建图像发布器
        image_publisher_ = image_transport_->advertise("image_raw", 10);
        camera_publisher_ = image_transport_->advertiseCamera("image", 10);
        
        // 注册参数回调
        parameters_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&HikCameraNode::on_parameters_changed, this, std::placeholders::_1));
    }

    void initialize_camera_info_manager()
    {
        camera_info_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(
            this, current_params_.camera_name, current_params_.camera_info_url);
        
        // 加载相机标定信息
        if (camera_info_manager_->validateURL(current_params_.camera_info_url)) {
            if (camera_info_manager_->loadCameraInfo(current_params_.camera_info_url)) {
                RCLCPP_INFO(this->get_logger(), "Loaded camera calibration from: %s", 
                           current_params_.camera_info_url.c_str());
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to load camera calibration from: %s", 
                           current_params_.camera_info_url.c_str());
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "Using default camera calibration");
        }
    }

    bool connect_camera()
    {
        RCLCPP_INFO(this->get_logger(), "Searching for cameras...");
        
        MV_CC_DEVICE_INFO_LIST device_list;
        memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        
        int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
        if (ret != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "EnumDevices failed: 0x%x", ret);
            return false;
        }
        
        if (device_list.nDeviceNum == 0) {
            RCLCPP_ERROR(this->get_logger(), "No cameras found");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Found %d camera(s)", device_list.nDeviceNum);

        // 选择第一个设备
        ret = MV_CC_CreateHandle(&camera_handle_, device_list.pDeviceInfo[0]);
        if (ret != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "CreateHandle failed: 0x%x", ret);
            return false;
        }

        ret = MV_CC_OpenDevice(camera_handle_);
        if (ret != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "OpenDevice failed: 0x%x", ret);
            MV_CC_DestroyHandle(camera_handle_);
            camera_handle_ = nullptr;
            return false;
        }

        is_connected_ = true;
        RCLCPP_INFO(this->get_logger(), "Camera connected successfully");
        return true;
    }

    bool configure_camera_parameters()
    {
        if (!is_connected_) return false;

        bool success = true;
        
        // 设置触发模式为连续采集
        int ret = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", MV_TRIGGER_MODE_OFF);
        if (ret != MV_OK) {
            log_camera_error(ret, "Set TriggerMode to Off");
            success = false;
        }

        // 设置分辨率
        ret = MV_CC_SetIntValueEx(camera_handle_, "Width", current_params_.width);
        if (ret != MV_OK) {
            log_camera_error(ret, "Set Width");
            success = false;
        }
        
        ret = MV_CC_SetIntValueEx(camera_handle_, "Height", current_params_.height);
        if (ret != MV_OK) {
            log_camera_error(ret, "Set Height");
            success = false;
        }

        // 设置其他参数
        ret = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", current_params_.exposure_time);
        if (ret != MV_OK) {
            log_camera_error(ret, "Set ExposureTime");
            success = false;
        }
        
        ret = MV_CC_SetFloatValue(camera_handle_, "Gain", current_params_.gain);
        if (ret != MV_OK) {
            log_camera_error(ret, "Set Gain");
            success = false;
        }
        
        ret = MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", current_params_.frame_rate);
        if (ret != MV_OK) {
            log_camera_error(ret, "Set AcquisitionFrameRate");
            success = false;
        }

        // 设置像素格式
        unsigned int pixel_format = pixel_format_string_to_enum(current_params_.pixel_format);
        if (pixel_format != 0) {
            ret = MV_CC_SetEnumValue(camera_handle_, "PixelFormat", pixel_format);
            if (ret != MV_OK) {
                log_camera_error(ret, "Set PixelFormat");
                success = false;
            }
        }

        // 分配图像缓冲区
        if (!allocate_image_buffer()) {
            success = false;
        }

        return success;
    }

    bool start_grabbing()
    {
        if (!is_connected_) return false;

        int ret = MV_CC_StartGrabbing(camera_handle_);
        if (ret != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "StartGrabbing failed: 0x%x", ret);
            return false;
        }

        is_grabbing_ = true;
        RCLCPP_INFO(this->get_logger(), "Image grabbing started");
        return true;
    }

    void stop_grabbing()
    {
        if (is_grabbing_ && camera_handle_) {
            MV_CC_StopGrabbing(camera_handle_);
            is_grabbing_ = false;
            RCLCPP_INFO(this->get_logger(), "Image grabbing stopped");
        }
    }

    void setup_timer()
    {
        // 根据帧率设置定时器周期
        double timer_period_ms = 1000.0 / current_params_.frame_rate;
        
        grab_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(timer_period_ms)),
            std::bind(&HikCameraNode::grab_and_publish_image, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "Image grab timer set to %.2f ms period", timer_period_ms);
    }

    void grab_and_publish_image()
    {
        if (!is_connected_ || !is_grabbing_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                               "Camera not ready, attempting reconnection...");
            attempt_reconnection();
            return;
        }

        MV_FRAME_OUT_INFO_EX frame_info;
        memset(&frame_info, 0, sizeof(MV_FRAME_OUT_INFO_EX));
        
        int ret = MV_CC_GetOneFrameTimeout(camera_handle_, image_buffer_.get(), 
                                         buffer_size_, &frame_info, 1000);

        if (ret == MV_OK) {
            publish_image(frame_info);
        } else {
            RCLCPP_ERROR(this->get_logger(), "GetOneFrameTimeout failed: 0x%x", ret);
            handle_camera_error();
        }
    }

    void publish_image(const MV_FRAME_OUT_INFO_EX& frame_info)
    {
        auto image_msg = std::make_unique<sensor_msgs::msg::Image>();
        auto camera_info_msg = std::make_unique<sensor_msgs::msg::CameraInfo>();
        
        // 设置消息头
        auto stamp = this->get_clock()->now();
        image_msg->header.stamp = stamp;
        image_msg->header.frame_id = current_params_.camera_frame_id;
        
        // 设置图像基本信息
        image_msg->height = frame_info.nHeight;
        image_msg->width = frame_info.nWidth;
        image_msg->data.resize(frame_info.nFrameLen);
        
        // 根据像素格式设置编码和步长
        std::string encoding;
        if (frame_info.enPixelType == PixelType_Gvsp_Mono8) {
            encoding = "mono8";
            image_msg->step = frame_info.nWidth;
        } else if (frame_info.enPixelType == PixelType_Gvsp_BGR8_Packed) {
            encoding = "bgr8";
            image_msg->step = frame_info.nWidth * 3;
        } else if (frame_info.enPixelType == PixelType_Gvsp_RGB8_Packed) {
            encoding = "rgb8";
            image_msg->step = frame_info.nWidth * 3;
        } else {
            RCLCPP_WARN_ONCE(this->get_logger(), 
                           "Unsupported pixel format: 0x%lx", frame_info.enPixelType);
            return;
        }
        
        image_msg->encoding = encoding;
        
        // 拷贝图像数据
        memcpy(image_msg->data.data(), image_buffer_.get(), frame_info.nFrameLen);
        
        // 获取相机信息
        *camera_info_msg = camera_info_manager_->getCameraInfo();
        camera_info_msg->header = image_msg->header;
        
        // 发布图像（带相机信息）
        camera_publisher_.publish(std::move(image_msg), std::move(camera_info_msg));
    }

    void attempt_reconnection()
    {
        cleanup();
        if (connect_camera() && configure_camera_parameters() && start_grabbing()) {
            RCLCPP_INFO(this->get_logger(), "Camera reconnected successfully");
        } else {
            RCLCPP_ERROR(this->get_logger(), "Camera reconnection failed");
        }
    }

    void handle_camera_error()
    {
        RCLCPP_ERROR(this->get_logger(), "Camera error detected, disconnecting...");
        is_connected_ = false;
        is_grabbing_ = false;
    }

    bool allocate_image_buffer()
    {
        MVCC_INTVALUE_EX payload_size;
        memset(&payload_size, 0, sizeof(MVCC_INTVALUE_EX));
        
        int ret = MV_CC_GetIntValueEx(camera_handle_, "PayloadSize", &payload_size);
        if (ret == MV_OK) {
            buffer_size_ = payload_size.nCurValue;
            image_buffer_ = std::make_unique<unsigned char[]>(buffer_size_);
            RCLCPP_INFO(this->get_logger(), "Image buffer allocated: %u bytes", buffer_size_);
            return true;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to get payload size: 0x%x", ret);
            return false;
        }
    }

    unsigned int pixel_format_string_to_enum(const std::string& format_str)
    {
        if (format_str == "mono8") return PixelType_Gvsp_Mono8;
        if (format_str == "bgr8") return PixelType_Gvsp_BGR8_Packed;
        if (format_str == "rgb8") return PixelType_Gvsp_RGB8_Packed;
        
        RCLCPP_ERROR(this->get_logger(), "Unsupported pixel format: %s", format_str.c_str());
        return 0;
    }

    bool set_resolution(int64_t width, int64_t height)
    {
        stop_grabbing();
        
        bool success = true;
        int ret = MV_CC_SetIntValueEx(camera_handle_, "Width", width);
        if (ret != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "Failed to set width: 0x%x", ret);
            success = false;
        }
        
        ret = MV_CC_SetIntValueEx(camera_handle_, "Height", height);
        if (ret != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "Failed to set height: 0x%x", ret);
            success = false;
        }
        
        if (success) {
            if (!allocate_image_buffer()) {
                success = false;
            } else {
                RCLCPP_INFO(this->get_logger(), "Resolution changed to %ldx%ld", width, height);
            }
        }
        
        start_grabbing();
        return success;
    }

    bool set_pixel_format(const std::string& format_str)
    {
        stop_grabbing();
        
        unsigned int pixel_format = pixel_format_string_to_enum(format_str);
        if (pixel_format == 0) {
            start_grabbing();
            return false;
        }
        
        int ret = MV_CC_SetEnumValue(camera_handle_, "PixelFormat", pixel_format);
        if (ret != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "Failed to set pixel format: 0x%x", ret);
            start_grabbing();
            return false;
        }
        
        start_grabbing();
        RCLCPP_INFO(this->get_logger(), "Pixel format changed to %s", format_str.c_str());
        return true;
    }

    rcl_interfaces::msg::SetParametersResult on_parameters_changed(
        const std::vector<rclcpp::Parameter>& parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";

        bool resolution_changed = false;
        bool pixel_format_changed = false;
        int64_t new_width = current_params_.width;
        int64_t new_height = current_params_.height;

        for (const auto& param : parameters) {
            const std::string& name = param.get_name();
            
            if (name == "exposure_time") {
                int ret = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", param.as_double());
                if (ret != MV_OK) {
                    result.successful = false;
                    result.reason = "Failed to set exposure time";
                } else {
                    current_params_.exposure_time = param.as_double();
                }
            }
            else if (name == "gain") {
                int ret = MV_CC_SetFloatValue(camera_handle_, "Gain", param.as_double());
                if (ret != MV_OK) {
                    result.successful = false;
                    result.reason = "Failed to set gain";
                } else {
                    current_params_.gain = param.as_double();
                }
            }
            else if (name == "frame_rate") {
                int ret = MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", param.as_double());
                if (ret != MV_OK) {
                    result.successful = false;
                    result.reason = "Failed to set frame rate";
                } else {
                    current_params_.frame_rate = param.as_double();
                    setup_timer(); // 重新设置定时器
                }
            }
            else if (name == "pixel_format") {
                pixel_format_changed = true;
                current_params_.pixel_format = param.as_string();
            }
            else if (name == "width") {
                resolution_changed = true;
                new_width = param.as_int();
                current_params_.width = new_width;
            }
            else if (name == "height") {
                resolution_changed = true;
                new_height = param.as_int();
                current_params_.height = new_height;
            }
            else if (name == "camera_name") {
                current_params_.camera_name = param.as_string();
                RCLCPP_INFO(this->get_logger(), "Camera name changed to: %s", 
                           current_params_.camera_name.c_str());
            }
            else if (name == "camera_frame_id") {
                current_params_.camera_frame_id = param.as_string();
                RCLCPP_INFO(this->get_logger(), "Camera frame ID changed to: %s", 
                           current_params_.camera_frame_id.c_str());
            }
        }

        // 处理需要重启采集的参数
        if (resolution_changed) {
            if (!set_resolution(new_width, new_height)) {
                result.successful = false;
                result.reason = "Failed to set resolution";
            }
        }
        
        if (pixel_format_changed) {
            if (!set_pixel_format(current_params_.pixel_format)) {
                result.successful = false;
                result.reason = "Failed to set pixel format";
            }
        }

        return result;
    }

    void log_camera_error(int error_code, const std::string& operation)
    {
        if (error_code != MV_OK) {
            RCLCPP_ERROR(this->get_logger(), "%s failed: 0x%x", operation.c_str(), error_code);
        }
    }

    void cleanup()
    {
        stop_grabbing();
        
        if (camera_handle_) {
            if (is_connected_) {
                MV_CC_CloseDevice(camera_handle_);
            }
            MV_CC_DestroyHandle(camera_handle_);
            camera_handle_ = nullptr;
        }
        
        is_connected_ = false;
        is_grabbing_ = false;
        image_buffer_.reset();
        buffer_size_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "Camera resources cleaned up");
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<HikCameraNode>();
        rclcpp::spin(node);
    }
    catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}