#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.h"

namespace MLLib {

struct Settings;

class Optimizer {
  public:
    Optimizer(Settings setting);
    virtual void update();
    virtual void reset();
    virtual void registerParams(Layer curLayer);
}

class SGD : Optimizer {
  public:
    SGD(Settings setting);

  private:
    Settings settings;
    std::vector<Tensor*> weights;
    std::vector<Tensor> velocity;
}

}
