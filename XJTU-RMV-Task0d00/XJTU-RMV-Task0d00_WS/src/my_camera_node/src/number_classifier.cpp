#include "number_classifier.hpp"
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <set>

NumberClassifier::NumberClassifier(
  rclcpp::Node * node, const std::string & model_path, const std::string & label_path,
  double confidence_threshold)
: node_(node), confidence_threshold_(confidence_threshold)
{
  // [!!!] 核心修改1：增加try-catch来捕获模型加载失败 [!!!]
  try {
    net_ = cv::dnn::readNetFromONNX(model_path);
    RCLCPP_INFO(node_->get_logger(), "ONNX model loaded successfully from: %s", model_path.c_str());
  } catch (const cv::Exception & e) {
    RCLCPP_FATAL(
      node_->get_logger(), "Failed to load ONNX model! Check model_path parameter. Error: %s",
      e.what());
    // 如果模型加载失败，我们应该让 net_ 保持为空，以便后续检查
    return;
  }

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
  // [!!!] 核心修改2：在开始处理前，检查网络是否有效 [!!!]
  if (net_.empty()) {
    RCLCPP_ERROR_ONCE(node_->get_logger(), "Classifier network is empty, skipping classification.");
    // 如果网络无效，我们选择最安全的策略：认为所有装甲板都无效，以避免后续错误
    armors.clear(); 
    return;
  }

  std::vector<Armor> classified_armors;
  static const std::set<std::string> valid_numbers = {"1", "2", "3", "4", "5"};

  const int warp_height = 28;
  const int warp_width = 28;
  const cv::Size roi_size(20, 28);

  for (auto & armor : armors) {
    // ... (透视变换逻辑保持不变)
    cv::Point2f lights_vertices[4] = {
      armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
      armor.right_light.bottom};

    cv::Point2f target_vertices[4] = {
      cv::Point(0, warp_height - 1),
      cv::Point(0, 0),
      cv::Point(warp_width - 1, 0),
      cv::Point(warp_width - 1, warp_height - 1),
    };

    cv::Mat number_image;
    auto rotation_matrix = cv::getPerspectiveTransform(lights_vertices, target_vertices);
    cv::warpPerspective(frame, number_image, rotation_matrix, cv::Size(warp_width, warp_height));

    // [!!!] 核心修改3：在裁剪ROI前，检查透视变换后的图像是否有效 [!!!]
    if (number_image.empty()) {
      RCLCPP_WARN(node_->get_logger(), "Warped image is empty, skipping this armor.");
      continue;
    }

    number_image = number_image(
      cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));
    
    // [!!!] 核心修改4：在送入网络前，检查最终的ROI是否有效 [!!!]
    if (number_image.empty()) {
      RCLCPP_WARN(node_->get_logger(), "Final ROI is empty, skipping this armor.");
      continue;
    }
    
    // ... (预处理和模型推理逻辑保持不变)
    cv::cvtColor(number_image, number_image, cv::COLOR_BGR2GRAY);
    cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat blob;
    cv::dnn::blobFromImage(number_image, blob, 1.0 / 255.0);
    net_.setInput(blob);
    cv::Mat outputs = net_.forward();
    
    // ... (解析和白名单筛选逻辑保持不变)
    double confidence;
    cv::Point class_id_point;
    minMaxLoc(outputs, nullptr, &confidence, nullptr, &class_id_point);
    
    if (confidence > confidence_threshold_) {
      std::string predicted_class = class_names_[class_id_point.x];
      if (valid_numbers.count(predicted_class)) {
        classified_armors.push_back(armor);
      }
    }
  }
  armors = classified_armors;
}