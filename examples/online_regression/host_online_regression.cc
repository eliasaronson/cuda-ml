#include "online_regression.h"
#include <iostream>

#define BACKWARD_HAS_BFD 1
#include <backward.hpp>
backward::SignalHandling sh;

int main() {
  ml::online_regression reg(0.001);

  // std::vector<double> x = {1, 2, 3, 11, 12, 13};
  // std::vector<double> w = {2, 4, 6};
  // std::vector<double> y = {28, 148};

  // std::vector<double> x = {1, 2, 3, 11, 12, 13, 21, 22, 23};
  // std::vector<double> w = {2, 4, 6};
  // std::vector<double> y = {172, 184, 196};

  // std::vector<double> x = {1,   2,   3,   11,   12,   13,   31,   32,   33,
  //                          0.1, 0.2, 0.3, 0.21, 0.22, 0.23, 0.31, 0.32,
  //                          0.33};
  // std::vector<double> w = {0.3, 0.2, 0.1};
  // std::vector<double> y = {1, 7, 13, 19, 0.1, 0.13};

  // std::vector<double> x = {1, 1, 1, 2, 2, 2, 2, 3};
  // std::vector<double> w = {1, 2};
  // std::vector<double> y = {6, 8, 9, 11};

  std::vector<double> x = {1, 2, 3};
  std::vector<double> w = {2};
  std::vector<double> y = {2, 4, 6};

  size_t X_features = 1;
  size_t Y_features = 1;

  printf("Partial solve\n");
  reg.partial_fit(x, X_features, y, Y_features);

  printf("Full solve\n");
  reg.fit(nullptr, nullptr, X_features, Y_features, x.size() / X_features);

  reg.predict(x, X_features, Y_features);
  double score = reg.score(x, X_features, y, Y_features);

  printf("score: %.10e\n", score);
}
