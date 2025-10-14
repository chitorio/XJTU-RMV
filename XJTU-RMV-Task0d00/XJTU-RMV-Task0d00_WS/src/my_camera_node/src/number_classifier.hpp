#ifndef NUMBER_CLASSIFIER_HPP_
#define NUMBER_CLASSIFIER_HPP_

#include "armor_types.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <rclcpp/rclcpp.hpp>

class NumberClassifier
{
public:
  NumberClassifier(
    rclcpp::Node * node, const std::string & model_path, const std::string & label_path,
    double confidence_threshold);

  void processArmors(const cv::Mat & frame, std::vector<Armor> & armors);

  /**
   * @brief 检测ROI中是否包含有效数字
   * @param number_roi 数字ROI图像
   * @param predicted_class 输出的预测类别
   * @param confidence 输出的置信度
   * @return 是否包含有效数字（1-5）
   */
  bool detectNumber(const cv::Mat& number_roi, std::string& predicted_class, double& confidence);

private:
  rclcpp::Node * node_;
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  double confidence_threshold_;
};

#endif // NUMBER_CLASSIFIER_HPP_