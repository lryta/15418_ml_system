#include <string>
#include <vector>
#include "tensor.h"
#include "operations/matrixOp.h"

namespace MLLib {

struct optimizerConfig {
  optimizerConfig(float lr,
      bool use_m = false, bool m = 0,
      bool use_w = false, bool w = 0): 
    lr(lr), use_monmentum(use_m), m(momentum),
    use_weightDecay(use_w), weightDecay(w)
  {}

  float lr; // learning rate
  bool use_monmentum;
  float momentum;
  bool use_weightDecay;
  float weightDecay;
}

class optimizer {
 public:
  optimizer(optimizerConfig setting);
  virtual void update();
  virtual void reset();
  virtual void registerParams(Layer curLayer);
}

class SGDOptimizer : optimizer {
 public:
  SGDOptimizer(optimizerConfig config_);

 private:
  optimizerConfig config_;
  std::vector<Tensor*> weights;
  std::vector<Tensor*> velocity;
}

}
