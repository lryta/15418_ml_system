#include "layer.h"

namespace MLLib {

class L2LossLayer:LossLayer{
public:
  L2LossLayer():LossLayer() {
  }

private:
  tensor weight_, bias_;
};


}
