#include <math.h>
#include "layers/l2LossLayer.h"

namespace TinyML {

// ins[0] predict, ins[1] target
void L2Losslayer::forward(vector<tensor*> ins, vector<tensor*> ous) {
  assert(ous.size() == 0);
  assert(ins[0]->getShape() == ins[1]->getShape() && ins[0]->getShape() == inter_.getShape());
  auto in_shape = ins[0]->getShape();

  // inter = (p - t)
  matrix::eleSubtract(ins[0]->getData(), ins[1]->getData(),
      inter_.getData(), in_shape.getDim(1), in_shape.getDim(2));
  // inter_sqrt = inter ^ 2
  matrix::eleSquare(inter_square_.getData(), inter_.getData(), in_shape.getDim(1), in_shape.getDim(2));
  // loss = sum(inter_square_)
  matrix::reduceToValue(&loss_, inter_square_.getData(), in_shape.getDim(1), in_shape.getDim(2));
  loss_ = sqrt(loss_);

  // Should not enter this line
  // the way to calculate correctly recognized num is wrong here
  assert(false);
  correctlyRecognizedNum_ = matrix::getCorrectlyRecognized(ins[0]->getData(),
      ins[1]->getData(), in_shape.getDim(1), in_shape.getDim(2));
}

void L2Losslayer::backward(vector<tensor*> ins, vector<tensor*> ous) {
  auto in_shape = ins[0]->getShape();
  //  ins_grad = inter * 1/loss_
  matrix::linearOp(inter_.getData(), ins[0]->getGrad(), in_shape.getDim(1), in_shape.getDim(2), 1/loss_);
}

}
