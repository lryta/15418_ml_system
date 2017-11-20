#include "layers/ActivationLayer.h"

namespace MLLib {

void SigmoidLayer::forward(vector<tensor> &ins, vector<tensor> &ous) {
  assert(ins.size() == 1);
  assert(ins.size() == ous.size());
  auto in_shape = ins[0].getShape();
  auto out_shape = ous[0].getShape();
  assert(in_shape == out_shape);
  matrix::sigmoidOp(ins[0].getData(), ous[0].getData(), in_shape.getDim(1), in_shape.getDim(2));
}

void SigmoidLayer::backward(vector<tensor> &ins, vector<tensor> &ous) {
  // inter_ = 1 - y
  matrix::linearOp(ous[0].getData(), inter_.getData(), in_shape.getDim(1), in_shape.getDim(2), -1, 1);
  // ins_grad = inter_ * y
  matrix::multiEle(ous[0].getData(), inter_.getData(), ins.getGrad(),
      in_shape.getDim(1), in_shape.getDim(2));
}

}
