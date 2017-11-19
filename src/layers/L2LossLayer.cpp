#include "L2LossLayer.h"

namespace MLLib {

// ins[0] predict, ins[1] target
void L2LossLayer::forward(vecotr<Tensor> &ins) {
  loss_ = 0;
  auto p_shape = ins[0].getShape();
  auto t_shape = ins[1].getShape();
  auto inter_shape = inter_.getShape();
  // inter = (p - t)
  matrix::subtract(ins[0].getData(), ins[1].getDelta(),
      inter_.getData(), ins[0].getDim(1), ins[0].getDim(2));
  // inter_sqrt = inter ^ 2
  matrix::square(inter_square_.getData(), inter_.getData(), ins[0].getDim(1), ins[0].getDim(2));
  // loss = sum(inter_square_)
  matrix::sum(&loss_, inter_square_.getData(), inter_shape.getDim(1), inter_shape.getDim(1));
  loss_ = std::sqrt(loss_);
}

void L2LossLayer::backward(vecotr<Tensor> &ins) {
  //  ins[0].delta = inter_data * 1/loss_
  matrix::multi(ins[0].getDelta(), 1/loss_, inter_.getData(), ins[0].getDim(1), ins[0].getDim(2));
}

}
