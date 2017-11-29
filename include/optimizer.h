#ifndef _TINYML_OPTIMIZER_H
#define _TINYML_OPTIMIZER_H

#include <string>
#include <vector>
#include "tensor.h"
#include "operations/matrixOp.h"

namespace TinyML {

struct optimizerConfig {
  optimizerConfig(float lr,
      bool use_m = false, bool m = 0,
      bool use_w = false, bool w = 0): 
    lr_(lr), use_monmentum_(use_m), use_weightDecay_(use_w),
    momentum_(m), weightDecay_(w) {}

  float lr_; // learning rate
  bool use_monmentum_, use_weightDecay_;
  float momentum_, weightDecay_;
};

class optimizer {
 public:
  optimizer();
  virtual void update();
  virtual void reset();
  virtual void registerParams(layer curlayer);
};

class SGDOptimizer : optimizer {
 public:
  SGDOptimizer(optimizerConfig *);

 private:
  optimizerConfig config_;
  std::vector<tensor*> weights_;
  std::vector<tensor*> velocity_;
};

}

#endif
