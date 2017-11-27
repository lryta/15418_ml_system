#include <math.h>
#include "L2LossLayer.h"

namespace TinyML {

// ins[0] predict, ins[1] target
void L2LossLayer::forward(vecotr<tensor> &ins, vecotr<tensor> &ous) {
  assert(ous.size() == 0);
  auto p_shape = ins[0].getShape();
  auto t_shape = ins[1].getShape();
  auto inter_shape = inter_.getShape();

  // inter = (p - t)
  matrix::eleSubtract(ins[0].getData(), ins[1].getData(),
      inter_.getData(), ins[0].getDim(1), ins[0].getDim(2));
  // inter_sqrt = inter ^ 2
  matrix::eleSquare(inter_square_.getData(), inter_.getData(), ins[0].getDim(1), ins[0].getDim(2));
  // loss = sum(inter_square_)
  matrix::reduceToValue(&loss_, inter_square_.getData(), inter_shape.getDim(1), inter_shape.getDim(1));
  loss_ = sqrt(loss_);
  correctlyRecognizedNum_ = matrix::eleSquare(ins[0].getData(), ins[1].getData());
}

void L2LossLayer::backward(vecotr<tensor> &ins, vecotr<tensor> &ous) {
  //  ins_grad = inter * 1/loss_
  matrix::linearOp(inter_.getData(), ins[0].getGrad(), ins[0].getDim(1), ins[0].getDim(2), 1/loss_);
}

}