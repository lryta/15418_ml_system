#ifndef _TINYML_OPTIMIZER_H
#define _TINYML_OPTIMIZER_H

#include <string>
#include <vector>
#include "tensor.h"
#include "layer.h"
#include "operations/matrixOp.h"

namespace TinyML {

struct optimizerConfig {
  optimizerConfig(float lr, float reg,
      bool use_m = false, bool m = 0,
      bool use_w = false, bool w = 0): 
    lr_(lr), reg_(reg), use_monmentum_(use_m), use_weightDecay_(use_w),
    momentum_(m), weightDecay_(w) {}

  float lr_; // learning rate
  float reg_; // ratio for regularization
  bool use_monmentum_, use_weightDecay_;
  float momentum_, weightDecay_;
};

class optimizer {
 public:
  optimizer() {}
  virtual void update() = 0;
  virtual void reset() = 0;
  virtual void registerParams(std::vector<layer*>,
    std::vector<tensor*>,
    std::vector<size_t>) = 0;
  virtual void randomizeParams() = 0;
};

class SGDOptimizer : public optimizer {
 public:
  SGDOptimizer(optimizerConfig *);
  virtual void update();
  virtual void reset();
  virtual void registerParams(std::vector<layer*>,
    std::vector<tensor*>,
    std::vector<size_t>);
  virtual void randomizeParams();

 private:
  optimizerConfig config_;
  std::vector<tensor*> weights_;
  std::vector<layer*> layers_;
  std::vector<size_t> param_id_map_layer_id_;
  std::vector<tensor*> velocity_;
};

}

#endif
