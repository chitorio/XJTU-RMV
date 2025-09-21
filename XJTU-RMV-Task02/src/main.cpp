#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

// ----- 工具函数部分 -----

// 转灰度
Mat convertToGray(const Mat& src) {
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    return gray;
}

// 转HSV
Mat convertToHSV(const Mat& src) {
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    return hsv;
}

// 均值滤波
Mat applyMeanBlur(const Mat& src, int ksize = 5) {
    Mat dst;
    blur(src, dst, Size(ksize, ksize));
    return dst;
}

// 高斯滤波
Mat applyGaussianBlur(const Mat& src, int ksize = 5) {
    Mat dst;
    GaussianBlur(src, dst, Size(ksize, ksize), 0);
    return dst;
}

// 提取红色区域
Mat extractRedRegions(const Mat& hsv) {
    Mat mask1, mask2, mask_red;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), mask1);
    inRange(hsv, Scalar(160, 100, 100), Scalar(180, 255, 255), mask2);
    bitwise_or(mask1, mask2, mask_red);
    return mask_red;
}

// 找到轮廓并绘制外轮廓和边界矩形
void findAndDrawContours(Mat& src, const Mat& mask) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        cout << "Contour " << i << " area: " << area << endl;

        // 绘制外轮廓
        drawContours(src, contours, (int)i, Scalar(0, 255, 0), 2);

        // 绘制边界矩形
        Rect boundingBox = boundingRect(contours[i]);
        rectangle(src, boundingBox, Scalar(255, 0, 0), 2);
    }
}

// 高亮处理（膨胀+腐蚀+漫水填充）
Mat highLightProcessing(const Mat& grey) {
    Mat bin;
    threshold(grey, bin, 128, 255, THRESH_BINARY);

    Mat dilated, eroded;
    dilate(bin, dilated, Mat(), Point(-1, -1), 2);
    erode(dilated, eroded, Mat(), Point(-1, -1), 2);

    Mat floodFilled = eroded.clone();
    floodFill(floodFilled, Point(0, 0), Scalar(255));

    return floodFilled;
}

// 绘制基本图形
void drawShapes(Mat& img) {
    circle(img, Point(50, 50), 30, Scalar(255, 0, 0), 2);
    rectangle(img, Rect(100, 100, 50, 50), Scalar(0, 255, 255), 2);
    putText(img, "shimingzi", Point(200, 200), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(255, 0, 255), 2);
}

// 图像旋转
Mat rotateImg(const Mat& src, double angle) {
    Point2f center(src.cols / 2.0, src.rows / 2.0);
    Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
    Mat dst;
    warpAffine(src, dst, rotMat, src.size());
    return dst;
}

// 图像裁减左上角1/4
Mat cropTopLeftQuarter(const Mat& src) {
    Rect roi(0, 0, src.cols / 2, src.rows / 2);
    return src(roi).clone();
}

// 返回以长边为基准的角度
static float normalizedAngle(const RotatedRect& r) {
    float w = r.size.width, h = r.size.height, ang = r.angle;
    if (w < h) ang += 90.0f;
    if (ang >= 180.0f) ang -= 180.0f;
    return ang;
}

// 检测装甲板
void detectLightBar(const Mat& src, Mat &dst) {
    dst = src.clone();

    // 图像灰度化
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    imwrite("output/img2_gray.png", gray);

    // 中值滤波
    Mat medianImg;
    medianBlur(gray, medianImg, 5);
    imwrite("output/img2_median.png", medianImg);

    // 自适应阈值处理
    Mat adaptive;
    adaptiveThreshold(medianImg, adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

    // hsv转换
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);
    // // 蓝色阈值
    // Scalar blueLow(100, 120, 100);
    // Scalar blueHigh(140, 255, 255);
    // Mat mask_blue;
    // inRange(hsv, blueLow, blueHigh, mask_blue);

    // // 白色阈值
    // Scalar whiteLow(0, 0, 240);
    // Scalar whiteHigh(150, 30, 255);
    // Mat mask_white;
    // inRange(hsv, whiteLow, whiteHigh, mask_white);

    // // Mat mask = mask_blue | mask_white;

    // Mat mask = mask_white;

    // 亮度掩码
    vector<Mat> hsvCh;
    split(hsv, hsvCh);
    Mat value = hsvCh[2];
    Mat brightMask;
    double brightThresh = 250;
    threshold(value, brightMask, brightThresh, 255, THRESH_BINARY);
    // mask = mask & brightMask;
    Mat mask = brightMask;

    imwrite("output/img2_beforemask.png", mask);

    // 形态学去噪（先开后闭）
    Mat dilateImg, erodeImg, morphImg;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    dilate(mask, dilateImg, kernel);
    erode(dilateImg, erodeImg, kernel);

    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    // 保存掩码
    imwrite("output/img2_mask.png", mask);

    // 找轮廓+筛选灯条
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<RotatedRect> lightBars;
    vector<float> angles;
    vector<float> areas;
    vector<float> ratios;

    Mat filteredContours = Mat::zeros(src.size(), CV_8UC3);

    for (auto& cont : contours) {
        double area = contourArea(cont);
        if (area < 500) continue; // 面积筛选

        // 椭圆拟合筛选
        if (cont.size() < 5) continue; // 点数太少无法拟合椭圆
        RotatedRect ellipseRect = fitEllipse(cont);

        float major = max(ellipseRect.size.width, ellipseRect.size.height);
        float minor = min(ellipseRect.size.width, ellipseRect.size.height);
        float ellipseRatio = major / minor;
        float ellipseArea = CV_PI * 0.25 * ellipseRect.size.width * ellipseRect.size.height;

        // 椭圆长宽比和面积筛选
        if (ellipseRatio < 1.2 || ellipseRatio > 10.0) continue;
        if (ellipseArea < 1300) continue;

        RotatedRect r = minAreaRect(cont);
        float w = r.size.width, h = r.size.height;
        float longSide = max(w, h), shortSide = min(w, h);
        if (shortSide < 20.0) continue; // 去掉过窄
        if (longSide > 200.0) continue; // 去掉过长

        float ratio = longSide / shortSide;
        if (ratio < 2.0 || ratio > 10.0) continue;  // 长宽比筛选
        lightBars.push_back(r);

        // 角度规范化
        float angle = normalizedAngle(r);

        lightBars.push_back(r);
        angles.push_back(angle);
        areas.push_back((float)area);
        ratios.push_back(ratio);

        vector<vector<Point>> singlecontour{cont};
        drawContours(filteredContours, singlecontour, -1, Scalar(0, 255, 0), 2);
    }
    imwrite("output/img2_filteredContours.png", filteredContours);

    // 配对灯条+框选装甲板
    for (size_t i = 0; i < lightBars.size(); i++) {
        for (size_t j = i + 1; j < lightBars.size(); j++) {
            // 角度接近
            if (fabs(angles[i] - angles[j]) > 10.0) continue;

            // 长宽比接近
            if (fabs(ratios[i] - ratios[j]) > 0.5) continue;

            // 面积接近
            if (fabs(areas[i] - areas[j]) > 50) continue;

            // 生成装甲板矩形
            Point2f center = (lightBars[i].center + lightBars[j].center) * 0.5f;
            float width = norm(lightBars[i].center - lightBars[j].center);
            float height = (lightBars[i].size.height + lightBars[j].size.height) * 0.5f;
            float angle = (angles[i] + angles[j]) * 0.5f;

            RotatedRect armorRect(center, Size2f(width, height), angle);

            // 画出装甲板
            Point2f pts[4];
            armorRect.points(pts);
            for (int k = 0; k < 4; k++) {
                line(dst, pts[k], pts[(k+1)%4], Scalar(0, 0, 255), 2);
            }
        }
    }
}

// ----- 主函数部分 -----

int main() {
    // 第一张图片处理
    Mat img1 = imread("resources/test_image.png");
    if (img1.empty()) {
        cerr << "Error: Could not open or find the image1!" << endl;
        return -1;
    }

    Mat gray1 = convertToGray(img1);
    imwrite("output/img1_grey.png", gray1);

    Mat hsv1 = convertToHSV(img1);
    imwrite("output/img1_hsv.png", hsv1);

    Mat meanBlur1 = applyMeanBlur(img1);
    imwrite("output/img1_meanBlur.png", meanBlur1);

    Mat gaussBlur1 = applyGaussianBlur(img1);
    imwrite("output/img1_gaussBlur.png", gaussBlur1);

    Mat redMask1 = extractRedRegions(hsv1);
    imwrite("output/img1_redMask.png", redMask1);

    Mat img1Contours = img1.clone();
    findAndDrawContours(img1Contours, redMask1);
    imwrite("output/img1_contours.png", img1Contours);

    Mat highLight1 = highLightProcessing(gray1);
    imwrite("output/img1_highlight.png", highLight1);

    Mat img1_shapes = img1.clone();
    drawShapes(img1_shapes);
    imwrite("output/img1_shapes.png", img1_shapes);

    Mat rotated1 = rotateImg(img1, 45);
    imwrite("output/img1_rotated.png", rotated1);

    Mat cropped1 = cropTopLeftQuarter(img1);
    imwrite("output/img1_cropped.png", cropped1);

    // 第二张图片处理，目标是提取蓝色灯条，并框选两条中间的矩形框
    Mat img2 = imread("resources/test_image_2.png");
    if (img2.empty()) {
        cerr << "Error: Could not open or find the image2!" << endl;
        return -1;
    }
    
    Mat dst;
    detectLightBar(img2, dst);

    imwrite("output/img2_result.png", dst);
    cout << "TARGET ACQUIRED!" << endl;

    return 0;
}