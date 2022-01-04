#include "../nanoflann.hpp"
#include <numeric>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include "stdlib.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <omp.h>
#include <chrono>


using namespace Eigen;
using namespace nanoflann;

typedef KDTreeEigenMatrixAdaptor<MatrixXd> my_kd_tree_t;

typedef struct NearestNeighbor {
  std::vector<int> closest_index;
  std::vector<double> closest_sqdistance;
} NearestNeighbor;

std::vector<double> load_pc(const char* fileName) {
  std::vector<double> d;
  std::string line;
  double str;
  std::stringstream ss;
  std::ifstream pc_data(fileName);
  while (getline(pc_data, line, '\n')) {
    ss.str(line);
    while (ss >> str) {
      d.push_back(str);
    }
    ss.clear();
  }
  return d;
}

void PCVector2Matrix(MatrixXd& mat, std::vector<double> pc_vector) {
#pragma omp parallel for
  for (size_t r = 0; r < mat.rows(); r++)
    for (size_t c = 0; c < mat.cols(); c++)
      mat(r, c) = pc_vector[mat.cols() * r + c];
}

NearestNeighbor getNearestNeighbors(const MatrixXd& mat,
                                    const my_kd_tree_t& my_tree) {
  NearestNeighbor nn;

  for (int i = 0; i < mat.rows(); ++i) {
    // Query point:
    std::vector<double> query_pt(mat.cols());
    for (size_t d = 0; d < mat.cols(); d++) {
      query_pt[d] = mat(i, d);
    }
    // do a knn search
    size_t ret_index;
    double out_dist_sqr;

    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_index, &out_dist_sqr);

    my_tree.index->findNeighbors(resultSet, &query_pt[0],
                                 nanoflann::SearchParams(10));

    nn.closest_index.push_back(static_cast<int>(ret_index));
    nn.closest_sqdistance.push_back(out_dist_sqr);
  }
  return nn;
}

Matrix4d find_optimal_rotation(const MatrixXd& pc_1, const MatrixXd& pc_2) {
  // http://nghiaho.com/?page_id=671

  Matrix4d trans_matrix = MatrixXd::Identity(4, 4);
  Vector3d centroid_1 = pc_1.colwise().mean();
  Vector3d centroid_2 = pc_2.colwise().mean();

  MatrixXd diff_pc_1 = pc_1.rowwise() - centroid_1.transpose();
  MatrixXd diff_pc_2 = pc_2.rowwise() - centroid_2.transpose();

  // covariance
  MatrixXd H = diff_pc_1.transpose() * diff_pc_2;
  MatrixXd U;
  MatrixXd V;
  Matrix3d R;
  Vector3d t;

  H /= diff_pc_1.rows();

  JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);
  U = svd.matrixU();
  V = svd.matrixV();

  R = V * U.transpose();

  // special reflection case, SVD result is not valid in real life
  if (R.determinant() < 0) {
    V.block<3, 1>(0, 2) *= -1;
    R = V * U.transpose();
  }

  t = centroid_2 - R * centroid_1;

  trans_matrix.block<3, 3>(0, 0) = R;
  trans_matrix.block<3, 1>(0, 3) = t;
  return trans_matrix;
}

Matrix3d eulerToRotationMatrix(double roll, double pitch, double yaw) {
  Matrix3d R_x;
  R_x << 1, 0, 0, 0, std::cos(roll), -std::sin(roll), 0, std::sin(roll),
      std::cos(roll);

  Matrix3d R_y;
  R_y << std::cos(pitch), 0, std::sin(pitch), 0, 1, 0, -std::sin(pitch), 0,
      std::cos(pitch);

  Matrix3d R_z;
  R_z << std::cos(yaw), -std::sin(yaw), 0, std::sin(yaw), std::cos(yaw), 0, 0,
      0, 1;

  return R_z * R_y * R_x;
}

Eigen::Quaterniond matrix2Quaternion(Eigen::Matrix3d m) {
  // matrix is orthogonal, "pure rotation"
  Eigen::Quaterniond q;

  q.w() = sqrt(1.0 + m(0, 0) + m(1, 1) + m(2, 2)) / 2.0;

  double w4 = (4.0 * q.w());
  q.x() = (m(2, 1) - m(1, 2)) / w4;
  q.y() = (m(0, 2) - m(2, 0)) / w4;
  q.z() = (m(1, 0) - m(0, 1)) / w4;

  return q;
}

Matrix4d ICP(MatrixXd pc_1, MatrixXd pc_2, int max_iteration,
             double threshold) {
  Matrix4d trans_matrix;
  trans_matrix.setIdentity();

  double prev_error = 0.0;
  double mean_error = 0.0;

  Matrix3d rotation_matrix = Matrix3d::Identity(3, 3);
  Vector3d translation_matrix = Vector3d::Zero(3, 1);

  my_kd_tree_t my_tree(3, std::cref(pc_2), 10 /* max leaf */);
  my_tree.index->buildIndex();

  int iter_break = 0;

  // Iterate either until
  // a. convergence (checked by threshold in change) or
  // b. max iteration number
  for (int i = 0; i < max_iteration; i++) {
    // Get closest points
    NearestNeighbor nn = getNearestNeighbors(pc_1, my_tree);
    MatrixXd target = MatrixXd::Zero(pc_1.rows(), 3);

    // 1. Pair points to closest
    for (int r = 0; r < pc_1.rows(); r++) {
      target(r, 0) = pc_2(nn.closest_index[r], 0);
      target(r, 1) = pc_2(nn.closest_index[r], 1);
      target(r, 2) = pc_2(nn.closest_index[r], 2);
    }

    // 2. Compute motion that minimises mean square error (MSE) between paired
    // points
    trans_matrix = find_optimal_rotation(pc_1, target);

    Matrix3d R = trans_matrix.block<3, 3>(0, 0);
    Vector3d TT = trans_matrix.block<3, 1>(0, 3);

    rotation_matrix *= R;
    translation_matrix += TT;

    // 3.a Apply motion to P
    for (int r = 0; r < pc_1.rows(); r++) {
      Vector3d point = Vector3d::Zero(3, 1);
      point(0, 0) = pc_1(r, 0);
      point(1, 0) = pc_1(r, 1);
      point(2, 0) = pc_1(r, 2);
      point = R * point;
      pc_1(r, 0) = point(0, 0);
      pc_1(r, 1) = point(1, 0);
      pc_1(r, 2) = point(2, 0);
    }

    pc_1 = pc_1.rowwise() + TT.transpose();

    // 3.b and update MSE
    mean_error = std::accumulate(nn.closest_sqdistance.begin(),
                                 nn.closest_sqdistance.end(), 0.0) /
                 nn.closest_sqdistance.size();

    if (abs(prev_error - mean_error) < threshold) {
      iter_break = i;
      break;
    }
    prev_error = mean_error;
  }

  trans_matrix.block<3, 3>(0, 0) = rotation_matrix;
  trans_matrix.block<3, 1>(0, 3) = translation_matrix;

  std::cout << "i: " << iter_break << std::endl;

  return trans_matrix;
}

// https://stackoverflow.com/questions/39693909/sort-eigen-matrix-column-values-by-ascending-order-of-column-1-values
bool compare_distance(const VectorXd& lhs, const VectorXd& rhs) {
  return lhs(2) < rhs(2);
}

MatrixXd sort_rows_by_distance(MatrixXd M) {
  std::vector<VectorXd> helper_vec;
  for (int64_t i = 0; i < M.rows(); ++i) helper_vec.push_back(M.row(i));

  std::sort(helper_vec.begin(), helper_vec.end(), &compare_distance);

  for (int64_t i = 0; i < M.rows(); ++i) M.row(i) = helper_vec[i];

  return M;
}

Matrix4d TrICP(MatrixXd mat1, MatrixXd mat2, int max_iteration,
               double threshold) {
  Matrix4d trans_matrix;
  trans_matrix.setIdentity();

  double prev_error = 0.0;
  double mean_error = 0.0;

  Matrix3d rotation_matrix = Matrix3d::Identity(3, 3);
  Vector3d translation_matrix = Vector3d::Zero(3, 1);

  my_kd_tree_t my_tree(3, std::cref(mat2), 10 /* max leaf */);
  my_tree.index->buildIndex();

  int NPo = 0.6 * mat1.rows();
  int iter_break = 0;

  for (int i = 0; i < max_iteration; i++) {
    MatrixXd sorted_by_dist = MatrixXd::Zero(mat1.rows(), 3);

    // 1. Closest point: For each point find closest point in M and compute sq
    // dist
    NearestNeighbor nn = getNearestNeighbors(mat1, my_tree);

    // 2. Trimmed Squares: sort by sq distance
    for (size_t r = 0; r < mat1.rows(); r++) {
      sorted_by_dist(r, 0) = r;
      sorted_by_dist(r, 1) = nn.closest_index[r];
      sorted_by_dist(r, 2) = nn.closest_sqdistance[r];
    }

    sorted_by_dist = sort_rows_by_distance(sorted_by_dist);

    MatrixXd new_mat1 = MatrixXd::Zero(NPo, 3);
    MatrixXd new_mat2 = MatrixXd::Zero(NPo, 3);

    // 2. Trimmed Squares: sort by sq distance
    for (int r = 0; r < NPo; r++) {
      new_mat1(r, 0) = mat1(sorted_by_dist(r, 0), 0);
      new_mat1(r, 1) = mat1(sorted_by_dist(r, 0), 1);
      new_mat1(r, 2) = mat1(sorted_by_dist(r, 0), 2);

      new_mat2(r, 0) = mat2(sorted_by_dist(r, 1), 0);
      new_mat2(r, 1) = mat2(sorted_by_dist(r, 1), 1);
      new_mat2(r, 2) = mat2(sorted_by_dist(r, 1), 2);
    }

    // 4. For Npo selected pairs compute optimal motion(R, t) that minimises
    // S_TS
    trans_matrix = find_optimal_rotation(new_mat1, new_mat2);

    Matrix3d R = trans_matrix.block<3, 3>(0, 0);
    Vector3d TT = trans_matrix.block<3, 1>(0, 3);

    rotation_matrix *= R;
    translation_matrix += TT;

    // 5. Data set motion
    for (int i = 0; i < mat1.rows(); i++) {
      Vector3d point = Vector3d::Zero(3, 1);
      point(0, 0) = mat1(i, 0);
      point(1, 0) = mat1(i, 1);
      point(2, 0) = mat1(i, 2);
      point = R * point;
      mat1(i, 0) = point(0, 0);
      mat1(i, 1) = point(1, 0);
      mat1(i, 2) = point(2, 0);
    }

    mat1 = mat1.rowwise() + TT.transpose();

    // Calculate trimmed MSE
    mean_error = sorted_by_dist.block(0, 2, NPo, 1).mean();

    // 3. Convergence test
    if (abs(prev_error - mean_error) < threshold) {
      iter_break = i;
      break;
    }
    prev_error = mean_error;
  }

  trans_matrix.block<3, 3>(0, 0) = rotation_matrix;
  trans_matrix.block<3, 1>(0, 3) = translation_matrix;
  std::cout << "i: " << iter_break << std::endl;

  return trans_matrix;
}

MatrixXd add_noise(MatrixXd pc, double noise_lvl) {
  int nr_of_noisy_pts = round(pc.rows() * noise_lvl);
  MatrixXd noisy_pc(pc.rows() + nr_of_noisy_pts, 3);

  // std::cout << nr_of_noisy_pts << std::endl;
  noisy_pc.block(0, 0, pc.rows(), 3) = pc;

  const double mean = 0.0;
  const double stddev = 0.5;
  auto dist = std::bind(std::normal_distribution<double>{mean, stddev},
                          std::mt19937(std::random_device{}()));

  double mean_x = pc.col(0).mean();
  double mean_y = pc.col(1).mean();
  double mean_z = pc.col(2).mean();

  for (int i = 0; i < nr_of_noisy_pts; i++) {
    noisy_pc(pc.rows() + i, 0) = mean_x + dist();
    noisy_pc(pc.rows() + i, 1) = mean_y + dist();
    noisy_pc(pc.rows() + i, 2) = mean_z + dist();
  }

  return noisy_pc;
}

int main(int argc, char** argv) {
  // input point cloud from http://graphics.stanford.edu/data/3Dscanrep/
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " POINT_CLOUD MAX_ITER THRESH NOISE_LVL" << std::endl;
    return 1;
  }

  int dims = 3;

  std::vector<double> pc_vector1 = load_pc(argv[1]);
  std::vector<double> pc_vector2 = load_pc(argv[5]);
  int max_iteration = atoi(argv[2]);
  double threshold = atof(argv[3]);
  double noise_lvl = atof(argv[4]);

  int row_count1 = static_cast<int>(pc_vector1.size() / dims);
  // int row_count2 = static_cast<int>(pc_vector2.size() / dims);

  MatrixXd from_pc(row_count1, dims);
  MatrixXd to_pc(row_count1, dims);
  Matrix4d transformation_matrix;
  PCVector2Matrix(from_pc, pc_vector1);
  // PCVector2Matrix(to_pc, pc_vector2);

  Matrix3d R;
  Vector3d TT(1.3, 0.5, 1);
  // // R = eulerToRotationMatrix(0.0349, 0.0174, 0);  // 2, 1, 0 degrees
  R = eulerToRotationMatrix(0.2617, 0.1745, 0.1745);  // 20, 10, 10 degrees
  Quaterniond Q = matrix2Quaternion(R);
  std::cout << R << std::endl;
  to_pc = from_pc * R;
  to_pc.rowwise() += TT.transpose();

  MatrixXd noisy_to_pc = add_noise(to_pc, noise_lvl);

  // std::cout << noisy_to_pc << std::endl;

  auto icp_start = std::chrono::high_resolution_clock::now();
  transformation_matrix = ICP(from_pc, noisy_to_pc, max_iteration, threshold);
  // transformation_matrix = ICP(from_pc, to_pc, max_iteration, threshold);
  auto icp_end = std::chrono::high_resolution_clock::now();
  auto icp_duration = std::chrono::duration_cast<std::chrono::duration<double>>(
      icp_end - icp_start);

  Matrix3d R_ICP = transformation_matrix.block<3, 3>(0, 0);
  Vector3d T_ICP = transformation_matrix.block<3, 1>(0, 3);

  Quaterniond Q_ICP = matrix2Quaternion(R_ICP);
  double icp_ang_dist = Q.angularDistance(Q_ICP);
  double icp_trans_mse = (TT - T_ICP).array().pow(2).sum() / 3;

  std::cout << "ICP took " << icp_duration.count() << " seconds" << std::endl;
  std::cout << "Transformation matrix: " << std::endl
            << transformation_matrix << std::endl
            << "angular distance: " << icp_ang_dist << std::endl
            << "translation error: " << icp_trans_mse << std::endl;

  auto tricp_start = std::chrono::high_resolution_clock::now();
  transformation_matrix = TrICP(from_pc, noisy_to_pc, max_iteration, threshold);
  auto tricp_end = std::chrono::high_resolution_clock::now();
  auto tricp_duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(tricp_end -
                                                                tricp_start);

  Matrix3d R_TrICP = transformation_matrix.block<3, 3>(0, 0);
  Vector3d T_TrICP = transformation_matrix.block<3, 1>(0, 3);

  Quaterniond Q_TrICP = matrix2Quaternion(R_TrICP);
  double tricp_ang_dist = Q.angularDistance(Q_TrICP);
  double tricp_trans_mse = (TT - T_TrICP).array().pow(2).sum() / 3;

  std::cout << "Trimmed ICP took " << tricp_duration.count() << " seconds"
            << std::endl;
  std::cout << "Transformation matrix: " << std::endl
            << transformation_matrix << std::endl
            << "angular distance: " << tricp_ang_dist << std::endl
            << "translation error: " << tricp_trans_mse << std::endl;

  return 0;
}
