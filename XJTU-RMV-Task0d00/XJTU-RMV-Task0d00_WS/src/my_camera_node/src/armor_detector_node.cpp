#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <rclcpp/parameter_event_handler.hpp>

using namespace std;
using namespace cv;

// CORRECTED: A helper function to safely scale a rectangle (ROI)
// scale_x, scale_y > 1.0 indicates enlargement
cv::Rect scale_rect(const cv::Rect &r, float scale_x, float scale_y) {
    // Explicitly cast r.tl() to cv::Point2f to resolve the type mismatch error
    cv::Point2f center = cv::Point2f(r.tl()) + cv::Point2f(r.width * 0.5f, r.height * 0.5f);
    float new_w = r.width * scale_x;
    float new_h = r.height * scale_y;
    return cv::Rect(
        static_cast<int>(center.x - new_w * 0.5f),
        static_cast<int>(center.y - new_h * 0.5f),
        static_cast<int>(new_w),
        static_cast<int>(new_h)
    );
}


class ArmorDetectorNode : public rclcpp::Node
{
public:
  ArmorDetectorNode() : Node("armor_detector_node")
  {
    subscription_ = image_transport::create_subscription(
      this, "image_raw", 
      std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    result_image_pub_ = image_transport::create_publisher(this, "armor_detector/result_image");
    
    debug_ = this->declare_parameter("debug", true);

    if (debug_) {
      createDebugPublishers();
    }
    
    param_subscriber_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    param_callback_handle_ = param_subscriber_->add_parameter_callback("debug",
      [this](const rclcpp::Parameter & p) {
        this->debug_ = p.as_bool();
        if (this->debug_) {
          this->createDebugPublishers();
        } else {
          this->destroyDebugPublishers();
        }
      });
      
    RCLCPP_INFO(this->get_logger(), "Armor Detector Node has been started.");
  }

private:
  // Core callback function with ROI logic
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    try {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      Mat frame = cv_ptr->image;
      
      Mat result_frame = frame.clone(); // Image for drawing the final result
      Mat processing_frame;             // The actual image being processed (full or ROI)
      cv::Point roi_offset(0, 0);       // Top-left offset of the ROI relative to the full image

      // ======================= ROI Logic [Start] =======================
      if (tracking_mode_) {
        // 1. In tracking mode, expand the previous ROI to define the current processing area
        cv::Rect potential_roi = scale_rect(last_armor_roi_, 2.0, 1.5);
        // 2. Ensure the ROI does not exceed the image boundaries
        last_armor_roi_ = potential_roi & cv::Rect(0, 0, frame.cols, frame.rows);
        
        processing_frame = frame(last_armor_roi_);
        roi_offset = last_armor_roi_.tl();
        
        // Draw the ROI area on the result image for debugging purposes
        cv::rectangle(result_frame, last_armor_roi_, cv::Scalar(0, 255, 255), 2);
      } else {
        // 3. If not in tracking mode, process the entire image
        processing_frame = frame;
      }
      // ======================= ROI Logic [End] =======================

      // Call the detection function, passing the processing image and the offset
      vector<RotatedRect> found_armors = detectLightBar(
          processing_frame, result_frame, msg->header, roi_offset);

      // ======================= Update Tracking State [Start] =======================
      if (!found_armors.empty()) {
        // If a target is found
        tracking_mode_ = true;
        lost_count_ = 0;
        // Update the ROI to be the bounding box of the first found armor
        last_armor_roi_ = found_armors[0].boundingRect();
        
        // Draw the found armors
        for (const auto& armor : found_armors) {
            Point2f pts[4];
            armor.points(pts);
            for (int k = 0; k < 4; k++) {
                line(result_frame, pts[k], pts[(k + 1) % 4], Scalar(0, 255, 0), 2);
            }
        }
      } else {
        // If no target is found
        lost_count_++;
        // If the target is lost for more than 10 consecutive frames, exit tracking mode and return to full-image search
        if (lost_count_ > 10) {
          tracking_mode_ = false;
        }
      }
      // ======================= Update Tracking State [End] =======================

      sensor_msgs::msg::Image::SharedPtr result_msg = 
        cv_bridge::CvImage(msg->header, "bgr8", result_frame).toImageMsg();
      result_image_pub_.publish(result_msg);
      
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  // Modified: The function returns a list of found armors and accepts an ROI offset
  // 修改：函数返回找到的装甲板列表，并接受ROI偏移量
  vector<RotatedRect> detectLightBar(const Mat& src, Mat &dst_to_draw_on, const std_msgs::msg::Header & header, const cv::Point & offset) {
      // ======================= 核心修改：采用灰度阈值法 [开始] =======================
      
      // 1. 将原图转换为灰度图
      Mat gray_img;
      cvtColor(src, gray_img, COLOR_BGR2GRAY);

      // 2. 根据亮度进行二值化
      // 这个阈值 `binary_thres` 是一个关键参数，需要根据实际场景亮度进行调整。
      // 可以先从 160 开始尝试。灯条越亮，这个值可以设得越高。
      Mat binary_img;
      int binary_thres = 160; 
      cv::threshold(gray_img, binary_img, binary_thres, 255, cv::THRESH_BINARY);
      
      // (可选) HSV方法被注释掉，作为对比保留
      // Mat hsv;
      // cvtColor(src, hsv, COLOR_BGR2HSV);
      // cv::Scalar lower_blue(100, 80, 80);
      // cv::Scalar upper_blue(130, 255, 255);
      // cv::inRange(hsv, lower_blue, upper_blue, binary_img);

      // 3. 形态学操作（现在对于更干净的二值图可能需要调整或省略）
      Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
      morphologyEx(binary_img, binary_img, MORPH_OPEN, kernel);

      // ======================= 核心修改：采用灰度阈值法 [结束] =======================


      if (debug_) {
        // 发布的调试图像现在是基于灰度阈值的结果
        auto mask_msg = cv_bridge::CvImage(header, "mono8", binary_img).toImageMsg();
        binary_mask_pub_.publish(std::move(mask_msg));
      }

      // 4. 寻找轮廓 (后续逻辑保持不变)
      vector<vector<Point>> contours;
      findContours(binary_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

      // 5. 筛选灯条
      vector<RotatedRect> lightBars;
      for (const auto& cont : contours) {
        double area = contourArea(cont);
        if (area < 15 || area > 500) continue;
        if (cont.size() < 5) continue;
        
        RotatedRect r = minAreaRect(cont);
        float longSide = max(r.size.width, r.size.height);
        float shortSide = min(r.size.width, r.size.height);
        if (shortSide < 1e-5) continue;
        
        float ratio = longSide / shortSide;
        if (ratio < 1.5 || ratio > 10.0) continue;
        
        lightBars.push_back(r);
      }
      
      vector<RotatedRect> found_armors; // 存储本函数找到的装甲板

      // 6. 配对灯条
      for (size_t i = 0; i < lightBars.size(); i++) {
          for (size_t j = i + 1; j < lightBars.size(); j++) {
              float angle_i = normalizedAngle(lightBars[i]);
              float angle_j = normalizedAngle(lightBars[j]);
              if (fabs(angle_i - angle_j) > 10.0) continue;
              
              float height_i = max(lightBars[i].size.width, lightBars[i].size.height);
              float height_j = max(lightBars[j].size.width, lightBars[j].size.height);
              if (fabs(height_i - height_j) / max(height_i, height_j) > 0.2) continue;
              
              Point2f center_diff = lightBars[i].center - lightBars[j].center;
              float distance = sqrt(center_diff.ddot(center_diff));
              float avg_height = (height_i + height_j) * 0.5;
              if(distance / avg_height < 1.0 || distance / avg_height > 5.0) continue;
              
              RotatedRect armorRect = RotatedRect(
                  (lightBars[i].center + lightBars[j].center) * 0.5f,
                  Size2f(distance, avg_height), 
                  (angle_i + angle_j) * 0.5f
              );
              
              // 将ROI内的坐标转换回全图坐标
              armorRect.center.x += offset.x;
              armorRect.center.y += offset.y;

              found_armors.push_back(armorRect);
          }
      }
      
      // 返回找到的装甲板列表
      return found_armors;
  }

  static float normalizedAngle(const RotatedRect& r) {
      float w = r.size.width, h = r.size.height, ang = r.angle;
      if (w < h) ang += 90.0f;
      if (ang >= 180.0f) ang -= 180.0f;
      return ang;
  }

  void createDebugPublishers() {
    if (!binary_mask_pub_) {
      binary_mask_pub_ = image_transport::create_publisher(this, "armor_detector/binary_mask");
      RCLCPP_INFO(this->get_logger(), "Debug mode enabled. Publishing binary mask topic.");
    }
  }

  void destroyDebugPublishers() {
    if (binary_mask_pub_) {
      binary_mask_pub_.shutdown();
      binary_mask_pub_ = {}; 
      RCLCPP_INFO(this->get_logger(), "Debug mode disabled. Stopped publishing binary mask topic.");
    }
  }

  // Member variables
  image_transport::Subscriber subscription_;
  image_transport::Publisher result_image_pub_;
  
  bool debug_;
  image_transport::Publisher binary_mask_pub_;
  std::shared_ptr<rclcpp::ParameterEventHandler> param_subscriber_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> param_callback_handle_;

  // State variables for ROI tracking
  bool tracking_mode_ = false;  // Is it in tracking mode?
  cv::Rect last_armor_roi_;     // ROI of the armor from the previous frame
  int lost_count_ = 0;          // Frame count of consecutive target loss
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmorDetectorNode>());
  rclcpp::shutdown();
  return 0;
}