#ifndef NUMBER_CLASSIFIER_HPP_
#define NUMBER_CLASSIFIER_HPP_

#include "armor_types.hpp" // [!!!] 修改点：包含新的类型定义文件
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

private:
  rclcpp::Node * node_;
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  double confidence_threshold_;
};

#endif // NUMBER_CLASSIFIER_HPP_