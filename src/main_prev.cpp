#include "../nanoflann.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>

#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
using namespace nanoflann;

const int SAMPLES_DIM = 15;

template <typename Der>
void generateRandomPointCloud(Eigen::MatrixBase<Der> &mat, const size_t N,
                              const size_t dim,
                              const typename Der::Scalar max_range = 10) {
  std::cout << "Generating " << N << " random points...";
  mat.resize(N, dim);
  for (size_t i = 0; i < N; i++)
    for (size_t d = 0; d < dim; d++)
      mat(i, d) = max_range * (rand() % 1000) / typename Der::Scalar(1000);
  std::cout << "done\n";
}

template <typename num_t>
void kdtree_demo(const size_t nSamples, const size_t dim) {
  Eigen::Matrix<num_t, Dynamic, Dynamic> mat(nSamples, dim);

  const num_t max_range = 20;

  // Generate points:
  generateRandomPointCloud(mat, nSamples, dim, max_range);

  //	cout << mat << endl;

  // Query point:
  std::vector<num_t> query_pt(dim);
  for (size_t d = 0; d < dim; d++)
    query_pt[d] = max_range * (rand() % 1000) / num_t(1000);

  // ------------------------------------------------------------
  // construct a kd-tree index:
  //    Some of the different possibilities (uncomment just one)
  // ------------------------------------------------------------
  // Dimensionality set at run-time (default: L2)
  // typedef KDTreeEigenMatrixAdaptor<Eigen::Matrix<num_t, Dynamic, Dynamic>>
  //    my_kd_tree_t;

  // Dimensionality set at compile-time
  //	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<num_t,Dynamic,Dynamic> >
  // my_kd_tree_t;

  // Dimensionality set at compile-time: Explicit selection of the distance
  // metric: L2
  typedef KDTreeEigenMatrixAdaptor<Eigen::Matrix<num_t, Dynamic, Dynamic>, -1,
                                   nanoflann::metric_L2>
      my_kd_tree_t;

  // Dimensionality set at compile-time: Explicit selection of the distance
  // metric: L2_simple
  //	typedef KDTreeEigenMatrixAdaptor<
  // Eigen::Matrix<num_t,Dynamic,Dynamic>,nanoflann::metric_L2_Simple>
  // my_kd_tree_t;

  // Dimensionality set at compile-time: Explicit selection of the distance
  // metric: L1
  //	typedef KDTreeEigenMatrixAdaptor<
  // Eigen::Matrix<num_t,Dynamic,Dynamic>,nanoflann::metric_L1>  my_kd_tree_t;

  my_kd_tree_t mat_index(dim, std::cref(mat), 10 /* max leaf */);
  mat_index.index->buildIndex();

  // do a knn search
  const size_t num_results = 1;
  vector<size_t> ret_indexes(num_results);
  vector<num_t> out_dists_sqr(num_results);

  nanoflann::KNNResultSet<num_t> resultSet(num_results);
  resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

  mat_index.index->findNeighbors(resultSet, &query_pt[0],
                                 nanoflann::SearchParams(10));

  std::cout << "knnSearch(nn=" << num_results << "): \n";
  for (size_t i = 0; i < num_results; i++)
    std::cout << "ret_index[" << i << "]=" << ret_indexes[i]
              << " out_dist_sqr=" << out_dists_sqr[i] << endl;
}

void cricle_demo() {
  using PointCloudType = Eigen::Matrix<float, Eigen::Dynamic, 2>;

  PointCloudType pc;
  pc.resize(10, Eigen::NoChange);  // nochange will result in 2

  const double r = 1.0;

  for (int i = 0; i < pc.rows(); i++) {
    double phi = static_cast<float>(i) / pc.rows() * 2 * 3.14159265;
    pc.row(i) = r * Eigen::Vector2f(std::cos(phi), std::sin(phi));
  }

  // generating kdtree
  typedef KDTreeEigenMatrixAdaptor<
      PointCloudType, PointCloudType::ColsAtCompileTime, nanoflann::metric_L2>
      my_kd_tree_t;

  my_kd_tree_t my_tree(PointCloudType::ColsAtCompileTime, std::cref(pc), 10);
  my_tree.index->buildIndex();

  
  nanoflann::KNNResultSet<float> result(1);
  size_t closest_index;
  float closest_sqdistance;
  
  // sample points from the line
  for (int i = 0; i < 5; i++){
    Eigen::Vector2f p(1, (static_cast<float>(i)/4) * 2 -1);
    result.init(&closest_index, &closest_sqdistance);

    my_tree.index->findNeighbors(result, p.data(), nanoflann::SearchParams());
    std::cout << closest_index << std::endl;
  }
  
  // for each, perform NN-search
}

int main(int argc, char **argv) {
  // Randomize Seed
  // srand(static_cast<unsigned int>(time(nullptr)));
  // kdtree_demo<float>(1000 /* samples */, SAMPLES_DIM /* dim */);

  cricle_demo();
  return 0;
}