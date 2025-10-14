#include "number_classifier.hpp"
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <set>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

NumberClassifier::NumberClassifier(
  rclcpp::Node * node, const std::string & model_path, const std::string & label_path,
  double confidence_threshold)
: node_(node), confidence_threshold_(confidence_threshold)
{
  // 1. 尝试加载ONNX模型，并在失败时提供清晰的错误信息
  try {
    net_ = cv::dnn::readNetFromONNX(model_path);
    RCLCPP_INFO(node_->get_logger(), "ONNX model loaded successfully from: %s", model_path.c_str());
  } catch (const cv::Exception & e) {
    RCLCPP_FATAL(
      node_->get_logger(), "Failed to load ONNX model! Check model_path parameter. Error: %s",
      e.what());
    // 如果模型加载失败，让 net_ 保持为空，以便后续检查
    return;
  }

  // 2. 加载标签文件
  std::ifstream label_file(label_path);
  if (!label_file.is_open()) {
    RCLCPP_FATAL(node_->get_logger(), "Failed to open label file! Check label_path parameter: %s", label_path.c_str());
    return;
  }
  std::string line;
  while (std::getline(label_file, line)) {
    class_names_.push_back(line);
  }

  RCLCPP_INFO(node_->get_logger(), "Number Classifier initialized successfully with %zu classes.", class_names_.size());
}

void NumberClassifier::processArmors(const cv::Mat & frame, std::vector<Armor> & armors)
{
  // 在开始处理前，检查网络是否有效
  if (net_.empty()) {
    RCLCPP_ERROR_ONCE(node_->get_logger(), "Classifier network is empty, skipping classification.");
    // 如果网络无效，最安全的策略是认为所有装甲板都无效，以避免后续错误
    armors.clear(); 
    return;
  }

  std::vector<Armor> classified_armors;
  static const std::set<std::string> valid_numbers = {"1", "2", "3", "4", "5"};

  const int warp_height = 28;
  const int warp_width = 28;
  const cv::Size roi_size(20, 28);

  for (auto & armor : armors) {
    // 1. 定义透视变换的源顶点和目标顶点
    cv::Point2f lights_vertices[4] = {
      armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
      armor.right_light.bottom};

    cv::Point2f target_vertices[4] = {
      cv::Point(0, warp_height - 1),
      cv::Point(0, 0),
      cv::Point(warp_width - 1, 0),
      cv::Point(warp_width - 1, warp_height - 1),
    };

    // 2. 进行透视变换
    cv::Mat number_image;
    auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
    cv::warpPerspective(frame, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

    // 3. 检查透视变换后的图像是否有效
    if (number_image.empty()) {
      RCLCPP_WARN(node_->get_logger(), "Warped image is empty, skipping this armor.");
      continue;
    }

    // 4. 提取数字ROI
    number_image = number_image(
      cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));
    
    // 5. 检查最终的ROI是否有效
    if (number_image.empty()) {
      RCLCPP_WARN(node_->get_logger(), "Final ROI is empty, skipping this armor.");
      continue;
    }
    
    // 6. 预处理ROI
    cv::Mat binary_roi;
    cv::cvtColor(number_image, binary_roi, cv::COLOR_BGR2GRAY);
    cv::threshold(binary_roi, binary_roi, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 7. 使用DNN模型进行分类
    cv::Mat blob;
    cv::dnn::blobFromImage(binary_roi, blob, 1.0 / 255.0);
    net_.setInput(blob);
    cv::Mat outputs = net_.forward();
    
    // 8. 解析结果并使用白名单进行筛选
    double confidence;
    cv::Point class_id_point;
    // 使用 softmax 获取更规范的置信度
    cv::Mat softmax_prob;
    cv::exp(outputs, softmax_prob);
    softmax_prob /= cv::sum(softmax_prob)[0];
    minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
    
    if (confidence > confidence_threshold_) {
      std::string predicted_class = class_names_[class_id_point.x];
      if (valid_numbers.count(predicted_class)) {
        classified_armors.push_back(armor);
      }
    }
  }
  // 用通过了分类器和白名单双重验证的装甲板列表，替换掉原来的列表
  armors = classified_armors;
}

bool NumberClassifier::detectNumber(const cv::Mat& number_roi, std::string& predicted_class, double& confidence)
{
  if (net_.empty()) {
    RCLCPP_ERROR_ONCE(node_->get_logger(), "Classifier network is empty, skipping number detection.");
    return false;
  }

  if (number_roi.empty()) {
    return false;
  }

  // 预处理ROI
  cv::Mat binary_roi;
  cv::cvtColor(number_roi, binary_roi, cv::COLOR_BGR2GRAY);
  cv::threshold(binary_roi, binary_roi, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  // 使用DNN模型进行分类
  cv::Mat blob;
  cv::dnn::blobFromImage(binary_roi, blob, 1.0 / 255.0);
  net_.setInput(blob);
  cv::Mat outputs = net_.forward();
  
  // 解析结果
  cv::Point class_id_point;
  cv::Mat softmax_prob;
  cv::exp(outputs, softmax_prob);
  softmax_prob /= cv::sum(softmax_prob)[0];
  minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
  
  predicted_class = class_names_[class_id_point.x];
  
  // 有效数字：1-5
  static const std::set<std::string> valid_numbers = {"1", "2", "3", "4", "5"};
  
  if (confidence > confidence_threshold_ && valid_numbers.count(predicted_class)) {
    return true;
  }
  
  return false;
}