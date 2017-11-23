#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.h"

namespace MLLib {

struct OptimizerConfig;

class Optimizer {
 public:
  Optimizer(OptimizerConfig setting);
  virtual void update();
  virtual void reset();
  virtual void registerParams(Layer curLayer);
}

class SGD : Optimizer {
 public:
  SGD(OptimizerConfig setting);

 private:
  Settings settings;
  std::vector<Tensor*> weights;
  std::vector<Tensor*> velocity;
}

}
