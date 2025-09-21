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

// 检测装甲板
void detectLightBar(const Mat& src, bool is_blue = true) {

    // 变量集中定义
    Mat channels[3], binary, Gaussian, dilatee;
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Rect boundRect;
    RotatedRect box;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<Point2f> boxPts(4);

    // 图像预处理
    split(src, channels);   // 分离通道
    threshold(channels[0], binary, 220, 255, 0); // 二值化
    GaussianBlur(binary, Gaussian, Size(5, 5), 0); // 高斯滤波
    dilate(Gaussian, dilatee, element); // 膨胀
    findContours(dilatee, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // 轮廓检测
    
    // 筛选灯条
    vector<float> widths, lengths, angles, areas;
    vector<Point2f> centers;

    for (int i = 0; i < contours.size(); i++) {
        // if (contours[i].empty()) continue;
        
        double area = contourArea(contours[i]);
        
        // 面积筛选
        if (area < 5 || contours[i].size() < 5) continue;

        // 椭圆拟合
        RotatedRect Light_Rec = fitEllipse(contours[i]);

        // 长宽比和面积比筛选
        if (Light_Rec.size.width / Light_Rec.size.height > 4) continue;
        
        widths.push_back(Light_Rec.size.width);
        lengths.push_back(Light_Rec.size.height);
        angles.push_back(Light_Rec.angle);
        areas.push_back(area);
        centers.push_back(Light_Rec.center);
    }

    // 灯条两两匹配
    for (size_t i = 0; i < centers.size(); i++) {
        for (size_t j = i + 1; j < centers.size(); j++) {
            float angleGap_ = abs(angles[i] - angles[j]);
            float LenGap_ratio = abs(lengths[i] - lengths[j]) / max(lengths[i], lengths[j]);
            float dis = sqrt(pow(centers[i].x - centers[j].x, 2) + pow(centers[i].y - centers[j].y, 2));
            float meanLen = (lengths[i] + lengths[j]) / 2;
            float lengap_ratio = (lengths[i] - lengths[j]) / meanLen;
            float yGap = abs(centers[i].y - centers[j].y);
            float yGap_ratio = yGap / meanLen;
            float xGap = abs(centers[i].x - centers[j].x);
            float xGap_ratio = xGap / meanLen;
            float ratio = dis / meanLen;

            // 匹配条件
            if (angleGap_ > 15 || LenGap_ratio > 1.0 || lengap_ratio > 0.8 || yGap_ratio > 1.5 || xGap_ratio > 2.2 || xGap_ratio < 0.8 || ratio > 3 || ratio < 0.8) {
                continue;
            }

            // 绘制匹配矩形
            Point center = Point((centers[i].x + centers[j].x) / 2, (centers[i].y + centers[j].y) / 2);
            RotatedRect rect = RotatedRect(center, Size(dis, meanLen), (angles[i] + angles[j]) / 2);
            Point2f vertices[4];
            rect.points(vertices);
            for (int k = 0; k < 4; k++) {
                line(src, vertices[k], vertices[(k + 1) % 4], Scalar(0, 255, 0), 2);
            }
        }
    }

    imwrite("output/img2_result.png", src);

    cout << "Detected " << centers.size() << " light bars." << endl;

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
    
    bool is_blue = true;
    detectLightBar(img2, is_blue);

    cout << "TARGET ACQUIRED!" << endl;

    return 0;
}