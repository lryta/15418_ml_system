#include "layer.h"
#include "shape.h"

namespace TinyML {

class ActivationLayer:Layer {
 public:
  ActivationLayer(vector<shape> &ins, vector<shape> &ous):
    Layer(ins, ous) {}

  // For activation layer, the output should be exactly the same
  void inferShape(vector<shape> &ins, vector<shape> &ous) {
    assert(ins.size() == 1);
    ous.clear();
    for (auto const& in : ins)
      ous.push_back(in);
  }

  // Usually no weight
  void initWeight(vector<shape> &ins, vector<shape> &ous) {}

  // Usually no weight
  vector<tensor&> getParam() { return {}; }
};

class SigmoidLayer:ActivationLayer {
 public:
  SigmoidLayer(vector<shape> &ins, vector<shape> &ous):
    ActivationLayer(ins, ous) { }

  void initIntermediateState(vector<shape> &ins, vector<shape> &ous)
  { inter_ = tensor(ins[0]); }
};

}
