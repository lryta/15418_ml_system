#include <math.h>
#include "negLogLikelilayer.h"

namespace TinyML {

// ins[0] predict, ins[1] target
void negLogLikelilayer::forward(vecotr<tensor> &ins, vecotr<tensor> &ous) {
  assert(ous.size() == 0);
  auto p_shape = ins[0].getShape();
  auto t_shape = ins[1].getShape();

  // predict = softmax(in)
  matrix::softmax(ins[0].getData(), predict_.getData(),
      ins[0].getDim(1), ins[0].getDim(2));

  // loss_ = neglog(predict, target)
  loss_ = matrix::negLogLikelihood(predict_.getData(), ins[1].getData(),
      ins[1].getDim(1), ins[1].getDim(2));

  correctlyRecognizedNum_ = matrix::getCorrectlyRecognized(predict_.getData(), ins[1].getData(),
      ins[0].getDim(1), ins[0].getDim(2));
}

void negLogLikelilayer::backward(vecotr<tensor> &ins, vecotr<tensor> &ous) {
  //  ins_grad = predict_ - target
  matrix::eleSubtract(predict_.getData(), ins[1].getData(), ins[0].getGrad(),
      ins[0].getDim(1), ins[0].getDim(2));
}

}
