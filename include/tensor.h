#include <stdlib.h>

class tensor {
public:
  // matrix M * N
  tensor(int m, int n, int k) {
    ptr_ = (float*) calloc(m * n * k * sizeof(float));
  }

private:
  float *ptr_;
};
