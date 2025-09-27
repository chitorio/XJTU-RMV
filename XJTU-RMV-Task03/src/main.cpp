#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

// ---------- 配置 ----------
const double VIDEO_FPS = 60.0;
const double G_MIN = 100.0;
const double G_MAX = 1000.0;
const double K_MIN = 0.01;
const double K_MAX = 1.0;
const int USE_FIRST_N_FRAMES_FOR_INIT = 8; // 用前8帧估计初值

// ---------- 工具函数 ----------
inline double sigmoid(double u) {
    return 1.0 / (1.0 + std::exp(-u));
}
inline double inv_sigmoid(double s) {
    return std::log(s / (1.0 - s));
}

// ---------- 自动微分成本函数 ----------
struct TrajResidual {
    TrajResidual(double t, double x_obs, double y_obs, double x0, double y0)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}

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
        T y_pred = T(y0_) + (vy + g / k) / k * (T(1.0) - exp_term) - g / k * t;

        residuals[0] = x_pred - T(x_obs_);
        residuals[1] = y_pred - T(y_obs_);
        return true;
    }

private:
    const double t_;
    const double x_obs_, y_obs_;
    const double x0_, y0_;
};

// ---------- 亮球检测 ----------
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
    if (contours.empty()) return false;

    int best_i = -1;
    double best_area = 0;
    for (size_t i=0; i<contours.size(); ++i) {
        double a = contourArea(contours[i]);
        if (a > best_area) { best_area = a; best_i = (int)i; }
    }
    if (best_i < 0) return false;
    Moments m = moments(contours[best_i]);
    if (m.m00 == 0) return false;

    cx = m.m10 / m.m00;
    cy = m.m01 / m.m00;
    return true;
}

// ---------- main函数 ----------
int main() {
    string video_path = "./TASK03/video.mp4";
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        cerr << "无法打开视频" << video_path << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS);
    cout << "视频 FPS: " << fps << endl;

    vector<double> times;
    vector<double> xs, ys;

    Mat frame;
    int frame_idx = 0;
    double x0 = 0, y0 = 0;
    bool have_x0 = false;

    while (cap.read(frame)) {
        double cx, cy;
        bool ok = detectProjectileCentroid(frame, cx, cy);
        if (!ok) { frame_idx++; continue; }

        double t = frame_idx / fps;
        if (!have_x0) {
            x0 = cx;
            y0 = cy;
            have_x0 = true;
            cout << "x0,y0 = (" << x0 << "," << y0 << ") \n";
        }

        times.push_back(t);
        xs.push_back(cx);
        ys.push_back(cy);
        frame_idx++;
    }

    if (times.size() < 6) { cerr << "读取帧数过少 (" << times.size() << ")\n"; return -1; }

    // 初值估计
    int n_init = min<int>(USE_FIRST_N_FRAMES_FOR_INIT, times.size()-1);
    double sum_vx = 0, sum_vy = 0;
    int count = 0;
    for (int i=0;i<n_init;i++) {
        double dt = times[i+1] - times[i];
        if (dt <= 0) continue;
        sum_vx += (xs[i+1] - xs[i]) / dt;
        sum_vy += (ys[i+1] - ys[i]) / dt;
        count++;
    }
    if (count == 0) count = 1;
    double vx0_init = sum_vx / count;
    double vy0_init = sum_vy / count;

    double g_mid = 0.5 * (G_MIN + G_MAX);
    double k_mid = 0.5 * (K_MIN + K_MAX);
    double u_g_init = inv_sigmoid((g_mid - G_MIN) / (G_MAX - G_MIN));
    double u_k_init = inv_sigmoid((k_mid - K_MIN) / (K_MAX - K_MIN));

    cout << "初始化预测: vx0=" << vx0_init << " px/s, vy0=" << vy0_init 
         << " px/s, g_mid=" << g_mid << ", k_mid=" << k_mid << endl;

    double vx0 = vx0_init;
    double vy0 = vy0_init;
    double u_g = u_g_init;
    double u_k = u_k_init;

    ceres::Problem problem;
    for (size_t i=0;i<times.size();++i) {
        ceres::CostFunction* cost = 
            new ceres::AutoDiffCostFunction<TrajResidual, 2, 1, 1, 1, 1>(
                new TrajResidual(times[i], xs[i], ys[i], x0, y0));
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
    cout << summary.FullReport() << endl;

    // 映射回实际参数
    double g_final = G_MIN + (G_MAX - G_MIN) * sigmoid(u_g);
    double k_final = K_MIN + (K_MAX - K_MIN) * sigmoid(u_k);

    cout << "修正后的结果\n";
    cout << "  vx0 = " << vx0 << " px/s\n";
    cout << "  vy0 = " << vy0 << " px/s\n";
    cout << "  g   = " << g_final << " px/s^2\n";
    cout << "  k   = " << k_final << " 1/s\n";

    // 计算RMSE
    double se = 0.0;
    for (size_t i=0; i<times.size(); ++i) {
        double t = times[i];
        double exp_term = std::exp(-k_final * t);
        double x_pred = x0 + vx0 / k_final * (1.0 - exp_term);
        double y_pred = y0 + (vy0 + g_final / k_final) / k_final * (1.0 - exp_term) - g_final / k_final * t;
        double dx = x_pred - xs[i];
        double dy = y_pred - ys[i];
        se += dx*dx + dy*dy;
    }
    double rmse = sqrt(se / (2.0*times.size()));
    cout << "RMSE: " << rmse << endl;

    // 预测轨迹存储
    vector<Point2d> preds;
    for (size_t i=0;i<times.size();++i) {
        double t = times[i];
        double exp_term = std::exp(-k_final * t);
        double x_pred = x0 + vx0 / k_final * (1.0 - exp_term);
        double y_pred = y0 + (vy0 + g_final / k_final) / k_final * (1.0 - exp_term) - g_final / k_final * t;
        preds.emplace_back(x_pred, y_pred);
    }

    cout << "frame_time, obs_x, obs_y, pred_x, pred_y\n";
    for (size_t i=0;i<times.size();++i) {
        printf("%.5f, %.3f, %.3f, %.3f, %.3f\n",
               times[i], xs[i], ys[i], preds[i].x, preds[i].y);
    }

    return 0;
}