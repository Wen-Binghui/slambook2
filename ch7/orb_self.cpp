//
// Created by xiang on 18-11-25.
//

#include <opencv2/opencv.hpp>
#include <string>
#include <nmmintrin.h>
#include <chrono>
#include <orb_pattern.h>

using namespace std;

// global variables
string first_file = "./1.png";
string second_file = "./2.png";

// 32 bit unsigned int, will have 8, 8x32=256
typedef vector<uint32_t> DescType; // Descriptor type

/**
 * compute descriptor of orb keypoints
 * @param img input image
 * @param keypoints detected fast keypoints
 * @param descriptors descriptors
 *
 * NOTE: if a keypoint goes outside the image boundary (8 pixels), descriptors will not be computed and will be left as
 * empty
 */
void ComputeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors);

/**
 * brute-force match two sets of descriptors
 * @param desc1 the first descriptor
 * @param desc2 the second descriptor
 * @param matches matches of two images
 */
void BfMatch(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<cv::DMatch> &matches);

int main(int argc, char **argv) {

  // load image
  cv::Mat first_image = cv::imread(first_file, 0);
  cv::Mat second_image = cv::imread(second_file, 0);
  assert(first_image.data != nullptr && second_image.data != nullptr);

  // detect FAST keypoints1 using threshold=40
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); //获取系统当前时间
  vector<cv::KeyPoint> keypoints1;
  cv::FAST(first_image, keypoints1, 40);
  vector<DescType> descriptor1;
  ComputeORB(first_image, keypoints1, descriptor1);
  
  // same for the second
  vector<cv::KeyPoint> keypoints2;
  vector<DescType> descriptor2;
  cv::FAST(second_image, keypoints2, 40);
  ComputeORB(second_image, keypoints2, descriptor2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  // find matches
  vector<cv::DMatch> matches;
  t1 = chrono::steady_clock::now();
  BfMatch(descriptor1, descriptor2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;
  cout << "matches: " << matches.size() << endl;

  // plot the matches
  cv::Mat image_show;
  cv::drawMatches(first_image, keypoints1, second_image, keypoints2, matches, image_show);
  cv::imshow("matches", image_show);
  cv::imwrite("matches.png", image_show);
  cv::waitKey(0);

  cout << "done." << endl;
  return 0;
}



// compute the descriptor
void ComputeORB(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, vector<DescType> &descriptors) {
  const int half_patch_size = 8;
  const int half_boundary = 16;
  int bad_points = 0;
  for (auto &kp: keypoints) {
    if (kp.pt.x < half_boundary || kp.pt.y < half_boundary ||
        kp.pt.x >= img.cols - half_boundary || kp.pt.y >= img.rows - half_boundary) {
      // outside
      bad_points++;
      descriptors.push_back({});
      continue;
    }

    float m01 = 0, m10 = 0;
    for (int dx = -half_patch_size; dx < half_patch_size; ++dx) {
      for (int dy = -half_patch_size; dy < half_patch_size; ++dy) {
        uchar pixel = img.at<uchar>(kp.pt.y + dy, kp.pt.x + dx);
        m10 += dx * pixel;
        m01 += dy * pixel;
      }
    }

    // angle should be arc tan(m01/m10);
    float m_sqrt = sqrt(m01 * m01 + m10 * m10) + 1e-18; // avoid divide by zero
    float sin_theta = m01 / m_sqrt;
    float cos_theta = m10 / m_sqrt;

    // compute the angle of this point
    DescType desc(8, 0);
    for (int i = 0; i < 8; i++) {
      uint32_t d = 0;
      for (int k = 0; k < 32; k++) {
        int idx_pq = i * 32 + k;
        cv::Point2f p(ORB_pattern[idx_pq * 4], ORB_pattern[idx_pq * 4 + 1]);
        cv::Point2f q(ORB_pattern[idx_pq * 4 + 2], ORB_pattern[idx_pq * 4 + 3]);

        // rotate with theta
        cv::Point2f pp = cv::Point2f(cos_theta * p.x - sin_theta * p.y, sin_theta * p.x + cos_theta * p.y)
                         + kp.pt;
        cv::Point2f qq = cv::Point2f(cos_theta * q.x - sin_theta * q.y, sin_theta * q.x + cos_theta * q.y)
                         + kp.pt;
        if (img.at<uchar>(pp.y, pp.x) < img.at<uchar>(qq.y, qq.x)) {
          d |= 1 << k;
        }
      }
      desc[i] = d;
    }
    descriptors.push_back(desc);
  }

  cout << "bad/total: " << bad_points << "/" << keypoints.size() << endl;
}

// brute-force matching
void BfMatch(const vector<DescType> &desc1, const vector<DescType> &desc2, vector<cv::DMatch> &matches) {
  const int d_max = 40;

  for (size_t i1 = 0; i1 < desc1.size(); ++i1) {
    if (desc1[i1].empty()) continue;
    cv::DMatch m{i1, 0, 256};
    for (size_t i2 = 0; i2 < desc2.size(); ++i2) {
      if (desc2[i2].empty()) continue;
      int distance = 0;
      for (int k = 0; k < 8; k++) {
        distance += _mm_popcnt_u32(desc1[i1][k] ^ desc2[i2][k]);
      }
      if (distance < d_max && distance < m.distance) {
        m.distance = distance;
        m.trainIdx = i2;
      }
    }
    if (m.distance < d_max) {
      matches.push_back(m);
    }
  }
}