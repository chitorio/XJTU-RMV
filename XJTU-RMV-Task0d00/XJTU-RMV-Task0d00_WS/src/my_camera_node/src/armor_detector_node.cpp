#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class ArmorDetectorNode : public rclcpp::Node
{
public:
  ArmorDetectorNode() : Node("armor_detector_node")
  {
    // 1. 创建订阅者，订阅原始图像话题
    // 使用image_transport可以高效处理图像消息
    subscription_ = image_transport::create_subscription(
      this, "image_raw", 
      std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    // 2. 创建发布者，用于发布带有识别结果的图像
    publisher_ = image_transport::create_publisher(this, "armor_detector/result_image");
    
    RCLCPP_INFO(this->get_logger(), "Armor Detector Node has been started.");
  }

private:
  // 核心回调函数，每当接收到新图像时执行
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    try {
      // 3. 将ROS图像消息转换为OpenCV图像格式
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
      Mat frame = cv_ptr->image;
      
      // 4. 调用你的装甲板检测函数
      Mat result_frame;
      detectLightBar(frame, result_frame); // 核心处理

      // 5. 将处理后的OpenCV图像转换回ROS消息并发布
      sensor_msgs::msg::Image::SharedPtr result_msg = 
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", result_frame).toImageMsg();
      publisher_.publish(result_msg);
      
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  // ----- 将你之前的识别代码移植到这里 -----
  
  // 返回以长边为基准的角度
  static float normalizedAngle(const RotatedRect& r) {
      float w = r.size.width, h = r.size.height, ang = r.angle;
      if (w < h) ang += 90.0f;
      if (ang >= 180.0f) ang -= 180.0f;
      return ang;
  }

  // 检测装甲板 (函数签名保持不变)
  void detectLightBar(const Mat& src, Mat &dst) {
      dst = src.clone();
      
      // 1. 将原图从 BGR 转换到 HSV 色彩空间
      Mat hsv;
      cvtColor(src, hsv, COLOR_BGR2HSV);

      // 2. 定义 蓝色 装甲板的HSV阈值范围
      // 注意：这些数值只是一个起点，你需要根据实际视频画面进行微调！
      cv::Scalar lower_blue(100, 80, 80);
      cv::Scalar upper_blue(130, 255, 255);

      // 如果要识别红色，需要两个范围
      // cv::Scalar lower_red1(0, 80, 80);
      // cv::Scalar upper_red1(10, 255, 255);
      // cv::Scalar lower_red2(170, 80, 80);
      // cv::Scalar upper_red2(180, 255, 255);
      // Mat mask1, mask2;
      // cv::inRange(hsv, lower_red1, upper_red1, mask1);
      // cv::inRange(hsv, lower_red2, upper_red2, mask2);
      // cv::bitwise_or(mask1, mask2, binary);


      // 3. 创建一个二值化的掩码（Mask）。只有在蓝色范围内的像素会变成白色。
      Mat binary;
      cv::inRange(hsv, lower_blue, upper_blue, binary);

      // 4. (可选但推荐) 使用形态学操作清理掩码中的噪点
      Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
      morphologyEx(binary, binary, MORPH_OPEN, kernel);

      // 形态学操作
      // Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // 也可以适当减小核
      // morphologyEx(binary, binary, MORPH_OPEN, kernel);
      
      vector<vector<Point>> contours;
      findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

      vector<RotatedRect> lightBars;
      for (const auto& cont : contours) {
        // --- 把最快、最有效的筛选放在最前面 ---
        
        // 1. 面积筛选: 快速剔除绝大部分噪声和无用的大块区域
        double area = contourArea(cont);
        if (area < 15 || area > 500) { // 增加一个面积上限！
            continue;
        }

        // 2. 轮廓点数筛选: fitEllipse 需要至少5个点
        if (cont.size() < 5) {
            continue;
        }

        // --- 现在才开始进行昂贵的计算 ---
        
        // 3. 外接矩形筛选: 灯条一定是瘦长的
        RotatedRect r = minAreaRect(cont);
        float longSide = max(r.size.width, r.size.height);
        float shortSide = min(r.size.width, r.size.height);
        
        // 避免除零错误
        if (shortSide < 1e-5) {
            continue;
        }

        float ratio = longSide / shortSide;
        // 设定一个严格的长宽比范围
        if (ratio < 1.5 || ratio > 10.0) {
            continue;
        }
        
        // 只有通过所有考验的，才是我们想要的灯条
        lightBars.push_back(r);
      }
      
      // 配对灯条
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

              // 距离与高度比筛选
              if(distance / avg_height < 1.0 || distance / avg_height > 5.0) continue;
              
              // 绘制装甲板
              Point2f pts[4];
              RotatedRect armorRect = RotatedRect((lightBars[i].center + lightBars[j].center) * 0.5f,
                                                 Size2f(distance, avg_height), 
                                                 (angle_i + angle_j) * 0.5f);
              armorRect.points(pts);
              for (int k = 0; k < 4; k++) {
                  line(dst, pts[k], pts[(k + 1) % 4], Scalar(0, 255, 0), 2);
              }
          }
      }
  }

  image_transport::Subscriber subscription_;
  image_transport::Publisher publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArmorDetectorNode>());
  rclcpp::shutdown();
  return 0;
}