#include <math.h>
#include "layers/negLogLikeliLayer.h"

namespace TinyML {

// ins[0] predict, ins[1] target
void negLogLikelilayer::forward(vector<tensor*> ins, vector<tensor*> ous) {
  assert(ous.size() == 0);
  assert(ins[0]->getShape() == ins[1]->getShape());
  auto in_shape = ins[0]->getShape();

  // predict = softmax(in)
  matrix::softmax(ins[0]->getData(), predict_.getData(),
      in_shape.getDim(1), in_shape.getDim(2));

  // loss_ = neglog(predict, target)
  loss_ = matrix::negLogLikelihood(predict_.getData(), ins[1]->getData(),
      in_shape.getDim(1), in_shape.getDim(2));

  correctlyRecognizedNum_ = matrix::getCorrectlyRecognized(predict_.getData(), ins[1]->getData(),
      in_shape.getDim(1), in_shape.getDim(2));
}

void negLogLikelilayer::backward(vector<tensor*> ins, vector<tensor*> ous) {
  //  ins_grad = predict_ - target
  auto in_shape = ins[0]->getShape();
  matrix::eleSubtract(predict_.getData(), ins[1]->getData(), ins[0]->getGrad(),
      in_shape.getDim(1), in_shape.getDim(2));
}

}
