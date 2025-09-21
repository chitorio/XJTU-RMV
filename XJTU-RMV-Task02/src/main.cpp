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
    Mat img;
    src.copyTo(img);

    // 颜色空间与扣图
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    Mat mask1, mask2, mask;

    if (is_blue) {
        inRange(hsv, Scalar(95, 80, 120), Scalar(135, 255, 255), mask);
    } else {
        inRange(hsv, Scalar(0, 90, 120), Scalar(10, 255, 255), mask1);
        inRange(hsv, Scalar(170, 90, 120), Scalar(180, 255, 255), mask2);
        mask = mask1 | mask2;
    }

    // 去小噪声：先开后闭
    Mat kernel_v = getStructuringElement(MORPH_RECT, Size(3, 9));
    morphologyEx(mask, mask, MORPH_OPEN, kernel_v, Point(-1, -1), 1);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel_v, Point(-1, -1), 1);

    // 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hier;
    findContours(mask, contours, hier, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    struct LightBar {
        RotatedRect rr;
        float len, wid, angle;  // angle归一到[0,90]
        Point2f center;
    }; vector<LightBar> bars;

    for (auto& c : contours) {
        if (c.size() < 5) continue;
        RotatedRect r = minAreaRect(c);
        float w = min(r.size.width, r.size.height);
        float h = max(r.size.width, r.size.height);
        float area = contourArea(c);
        if (area < 30) continue;
        float ratio = h / max(w, 1.0f);
        if (ratio < 3.0f || h < 8) continue;
        float ang = r.angle;
        if (r.size.height >= r.size.width) {
            ang = abs(ang + 90.f);
        } else {
            ang = abs(ang);
        }
        if (ang > 25) continue;

        bars.push_back({r, h, w, ang, r.center});
    }

    // 配对+打分
    struct PairCand {
        int i, j;
        float score;
        RotatedRect armor;
    }; vector<PairCand> pairs;

    auto armorForm2 = [](const LightBar& a, const LightBar& b) {
        Point2f c((a.center.x + b.center.x) / 2.f, (a.center.y + b.center.y) / 2.f);
        float meanLen = 0.5f * (a.len + b.len);
        float meanWid = 0.5f * (a.wid + b.wid);
        float dx = abs(a.center.x - b.center.x);
        float dy = abs(a.center.y - b.center.y);
        float angle = 0.5f * (a.rr.angle + b.rr.angle);

        if (angle < -90) angle += 180;
        if (angle > 90) angle -= 180;
        float width = dx + meanWid;
        float height = meanLen;
        return RotatedRect(c, Size2f(width, height), angle);
    };

    for (int i = 0; i < (int)bars.size(); ++i) {
        for (int j = i + 1; j < (int)bars.size(); ++j) {
            const auto& A = bars[i], & B = bars[j];
            float len_sim = abs(A.len - B.len) / max(A.len, B.len);
            float ang_sim = abs(A.angle - B.angle);
            float dx = abs(A.center.x - B.center.x);
            float dy = abs(A.center.y - B.center.y);
            float meanLen = 0.5f * (A.len + B.len);
            float ratio = dx / max(meanLen, 1.0f);

            // 几何先验
            if (len_sim > 0.35f) continue;
            if (ang_sim > 12.0f) continue;
            if (dy/meanLen > 0.4f) continue;
            if (ratio < 0.3f || ratio > 3.5f) continue;

            // 打分
            float score = 2.0f*len_sim + 0.5f*(ang_sim/12.0f) + 0.8f*(dy/meanLen) + 0.6f*abs(ratio-1.2f);
            pairs.push_back({i, j, score, armorForm2(A, B)});
        }
    }

    // 选分最低的
    sort(pairs.begin(), pairs.end(), [](auto& a, auto& b){return a.score < b.score;});
    vector<RotatedRect> armors;
    for (auto& p : pairs) {
        bool keep = true;
        for (auto& q : armors) {
            Point2f pc = p.armor.center, qc = q.center;
            if (norm(pc - qc) < 0.5f*(p.armor.size.width + q.size.width)) {
                keep =false;
                break;
            }
        }
        if (keep) armors.push_back(p.armor);
    }

    // 可视化
    for (auto& b : bars) {
        Point2f v[4];
        b.rr.points(v);
        for (int k = 0; k < 4; ++k) line(img, v[k], v[(k+1)%4], Scalar(0, 255, 255), 2);
    }
    for (auto& r : armors) {
        Point2f v[4];
        r.points(v);
        for (int k = 0; k < 4; ++k) line(img, v[k], v[(k+1)%4], Scalar(0, 255, 0), 3);
    }

    imwrite("output/img2_result.png", img);
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