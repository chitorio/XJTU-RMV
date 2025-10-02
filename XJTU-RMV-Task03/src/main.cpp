#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
using namespace cv;
const double VIDEO_FPS = 60.0;
const double G_MIN = 100.0;
const double G_MAX = 1000.0;
const double K_MIN = 0.01;
const double K_MAX = 1.0;
const int USE_FIRST_N_FRAMES_FOR_INIT = 8;
inline double sigmoid(double u) {
    return 1.0 / (1.0 + std::exp(-u));
}
inline double inv_sigmoid(double s) {
    return std::log(s / (1.0 - s));
}
struct TrajResidual {
    TrajResidual(double t, double x_obs, double y_obs, double x0, double y0, int img_height)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0), img_height_(img_height) {}
    template <typename T>
    bool operator()(const T* const vx0,
                    const T* const vy0,
                    const T* const u_g,
                    const T* const u_k,
                    T* residuals) const {
        T s_g = T(1.0) / (T(1.0) + exp(-u_g[0]));
        T s_k = T(1.0) / (T(1.0) + exp(-u_k[0]));
        T g = T(G_MIN) + (T(G_MAX) - T(G_MIN)) * s_g;
        T k = T(K_MIN) + (T(K_MAX) - T(K_MIN)) * s_k;
        T t = T(t_);
        T vx = vx0[0];
        T vy = vy0[0];
        T exp_term = ceres::exp(-k * t);        
        T x_pred = T(x0_) + vx / k * (T(1.0) - exp_term);
        T y_pred_physics = T(y0_) + (vy + g / k) / k * (T(1.0) - exp_term) - g / k * t;
        T y_pred_image = T(img_height_) - y_pred_physics;
        residuals[0] = x_pred - T(x_obs_);
        residuals[1] = y_pred_image - T(y_obs_);
        return true;
    }
private:
    const double t_;
    const double x_obs_, y_obs_;
    const double x0_, y0_;
    const int img_height_;
};
bool detectProjectileCentroid(const Mat& frame, double &cx, double &cy) {
    Mat gray;
    if (frame.channels() == 3) cvtColor(frame, gray, COLOR_BGR2GRAY);
    else gray = frame;
    GaussianBlur(gray, gray, Size(5,5), 0);
    Mat th;
    double thresh_val = 150; 
    threshold(gray, th, thresh_val, 255, THRESH_BINARY);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(th, th, MORPH_OPEN, kernel);
    vector<vector<Point>> contours;
    findContours(th, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int best_i = -1;
    double best_area = 0;
    for (size_t i=0; i<contours.size(); ++i) {
        double a = contourArea(contours[i]);
        if (a > best_area) { best_area = a; best_i = (int)i; }
    }
    Moments m = moments(contours[best_i]);
    cx = m.m10 / m.m00;
    cy = m.m01 / m.m00;
    return true;
}
int main() {
    string video_path = "./TASK03/video.mp4";
    VideoCapture cap(video_path);
    double fps = cap.get(CAP_PROP_FPS);
    Mat first_frame;
    cap.read(first_frame);
    cap.set(CAP_PROP_POS_FRAMES, 0);
    int img_height = first_frame.rows;
    vector<double> times;
    vector<double> xs, ys;
    Mat frame;
    int frame_idx = 0;
    double x0_img = 0, y0_img = 0;
    bool have_x0 = false;
    while (cap.read(frame)) {
        double cx, cy;
        bool ok = detectProjectileCentroid(frame, cx, cy);
        if (!ok) { frame_idx++; continue; }
        double t = frame_idx / fps;
        if (!have_x0) {
            x0_img = cx;
            y0_img = cy;
            have_x0 = true;
        }
        times.push_back(t);
        xs.push_back(cx);      // 图像坐标系x
        ys.push_back(cy);      // 图像坐标系y
        frame_idx++;
    }
    double x0_physics = x0_img;
    double y0_physics = img_height - y0_img;
    int n_init = min<int>(USE_FIRST_N_FRAMES_FOR_INIT, times.size()-1);
    double sum_vx = 0, sum_vy = 0;
    int count = 0;
    for (int i=0;i<n_init;i++) {
        double dt = times[i+1] - times[i];
        if (dt <= 0) continue;        
        double x1_img = xs[i+1], y1_img = ys[i+1];
        double x0_img = xs[i], y0_img = ys[i];        
        double x1_physics = x1_img;
        double y1_physics = img_height - y1_img;
        double x0_physics = x0_img;
        double y0_physics = img_height - y0_img;        
        sum_vx += (x1_physics - x0_physics) / dt;
        sum_vy += (y1_physics - y0_physics) / dt;
        count++;
    }
    if (count == 0) count = 1;
    double vx0_init = sum_vx / count;
    double vy0_init = sum_vy / count;
    double g_mid = 0.5 * (G_MIN + G_MAX);
    double k_mid = 0.5 * (K_MIN + K_MAX);
    double u_g_init = inv_sigmoid((g_mid - G_MIN) / (G_MAX - G_MIN));
    double u_k_init = inv_sigmoid((k_mid - K_MIN) / (K_MAX - K_MIN));
    double vx0 = vx0_init;
    double vy0 = vy0_init;
    double u_g = u_g_init;
    double u_k = u_k_init;
    ceres::Problem problem;
    for (size_t i=0;i<times.size();++i) {
        ceres::CostFunction* cost = 
            new ceres::AutoDiffCostFunction<TrajResidual, 2, 1, 1, 1, 1>(
                new TrajResidual(times[i], xs[i], ys[i], x0_physics, y0_physics, img_height));
        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost, loss, &vx0, &vy0, &u_g, &u_k);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    options.num_threads = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    double g_final = G_MIN + (G_MAX - G_MIN) * sigmoid(u_g);
    double k_final = K_MIN + (K_MAX - K_MIN) * sigmoid(u_k);
    cout << "=== 拟合结果（物理坐标系）===" << endl;
    cout << "  vx0 = " << vx0 << " px/s\n";
    cout << "  vy0 = " << vy0 << " px/s\n";
    cout << "  g   = " << g_final << " px/s^2\n";
    cout << "  k   = " << k_final << " 1/s\n";
    return 0;
}