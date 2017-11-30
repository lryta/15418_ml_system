#include <chrono>
#include "weightInit.h"

namespace TinyML {

void weightInit::uniformInit(tensor *t, float low_thresh, float high_thresh) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> dis(low_thresh, high_thresh);
  size_t total = t->getShape().getTotal();
  float *data = t->getData();
  for (size_t i = 0; i < total; ++i)
    data[i] = dis(generator);
}

void weightInit::normalDistInit(tensor *t, float mean, float stdv) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<float> dis(mean, stdv);
  size_t total = t->getShape().getTotal();
  float *data = t->getData();
  for (size_t i = 0; i < total; ++i)
    data[i] = dis(generator);
}

}
