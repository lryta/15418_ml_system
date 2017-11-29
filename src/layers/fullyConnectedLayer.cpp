#include "layers/FullyConnectedlayer.h"
#include "MatrixOp.h"

namespace TinyML {

vector<tensor&> inferShape::getParam() {
  return {weight_, bias_};
}

vector<tensor&> FullyConnectedlayer::inferShape(vector<tensor> &ins, vector<tensor> &ous) {
  auto in_shape = ins[0];
  ous.clear();
  ous.push_back(tensor(in_shape.getDim(1), inter_dim_));
}

vector<tensor&> FullyConnectedlayer::initWeight(vector<tensor> &ins, vector<tensor> &ous) {
  weight_ = tensor({ins.getDim(2), inter_dim_});
  bias_ = tensor({inter_dim_});
}

void FullyConnectedlayer::forward(vector<tensor> &ins, vector<tensor> &ous) {
  assert(ins.size() == 1);
  assert(ous.size() == 1);
  auto in_shape = ins[0].getShape();
  auto out_shape = ous[0].getShape();
  auto bias_shape = bias_.getShape();

  assert(in_shape.getDim(2) == weight_.getDim(1));
  assert(bias_shape.getDim(1) == weight_.getDim(2));

  // y = x * w + b
  matrix::gemm(ins[0].getData(), weight_.getData(), bias_.getData(), ous[0].getData(),
      out_shape.getDim(1), in_shape.getDim(2), out_shape.getDim(2));
}

void FullyConnectedlayer::backward(vector<tensor> &ins, vector<tensor> &ous) {
  auto in_shape = ins[0].getShape();
  auto w_shape = weight_.getShape();
  auto out_shape = ous[0].getShape();

  // Get Delta of ins
  // x_grad = y_grad * trans(w)
  assert(out_shape.getDim(2) == w_shape.getDim(2));
  matrix::gemm(ous[0].getGrad(), weight_.getData(), NULL, ins[0].getGrad(),
      in_shape.getDim(1), out_shape.getDim(2), in_shape.getDim(2), false, true);

  // Get Delta of w
  // w_grad = trans(x) * y_grad
  assert(in_shape.getDim(1) == out_shape.getDim(1));
  matrix::gemm(ins[0].getData(), ous[0].getGrad(), NULL, weight_.getGrad(),
      w_shape.getDim(1), in_shape.getDim(1), w_shape.getDim(2), true, false);

  // Get Delta of b
  // bias_grad = sum(y_grad, 1)
  matrix::reduceMatrix(ous[0].getGrad(), bias_.getGrad(), out_shape.getDim(1),  out_shape.getDim(2), 1);
}

};
