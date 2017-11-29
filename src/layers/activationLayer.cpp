#include "layers/activationLayer.h"
#include "operations/matrixOp.h"

namespace TinyML {

void Sigmoidlayer::forward(vector<tensor*> ins, vector<tensor*> ous) {
  assert(ins.size() == 1);
  assert(ins.size() == ous.size());
  auto in_shape = ins[0]->getShape();
  auto out_shape = ous[0]->getShape();
  assert(in_shape == out_shape);
  matrix::sigmoidOp(ins[0]->getData(), ous[0]->getData(), in_shape.getDim(1), in_shape.getDim(2));
}

void Sigmoidlayer::backward(vector<tensor*> ins, vector<tensor*> ous) {
  auto in_shape = ins[0]->getShape();
  // inter_ = 1 - y
  matrix::linearOp(ous[0]->getData(), inter_.getData(),
      in_shape.getDim(1), in_shape.getDim(2), -1, 1);

  // inter_ = inter_ * y
  matrix::multiEleInplace(ous[0]->getData(), inter_.getData(),
      in_shape.getDim(1), in_shape.getDim(2));

  // in_grad = out_grad * y * (1-y)
  matrix::multiEle(ous[0]->getGrad(), inter_.getData(), ins[0]->getGrad(),
      in_shape.getDim(1), in_shape.getDim(2));
}

}
