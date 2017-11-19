#include "layers/FullyConnectedLayer.h"
#include "MatrixOp.h"

namespace MLLib {

void FullyConnectedLayer::forward(vector<tensor> &ins, vector<tensor> &ous) {
  assert(ins.size() == 1);
  auto in_shape = ins[0].getShape();
  auto w_shape = weight_.getShape();

  // data_num = in_shape.getDim(1);
  // feature_dim = in_shape.getDim(2);
  // w_col = w_shape.getDim(2);

  assert(in_shape.getDim(2) == weight_.getDim(1));
  shape out_shape(in_shape.getDim(1),  w_shape.getDim(2)) ;

  ous.emplace_back(out_shape);
  matrix::gemm(ins[0].getData(), weight_.getData(), bias_.getData(), ous[0].getData(),
      out_shape.getDim(1), in_shape.getDim(2), out_shape.getDim(2));
}

void FullyConnectedLayer::backward(vector<tensor> &ins, vector<tensor> &ous) {
  auto in_shape = ins[0].getShape();
  auto w_shape = weight_.getShape();
  auto out_shape = ous[0].getShape();

  // Get Delta of ins
  assert(out_shape.getDim(2) == w_shape.getDim(2));
  matrix::gemm(ous[0].getGrad(), weight_.getData(), NULL, ins[0].getGrad(),
      in_shape.getDim(1), out_shape.getDim(2), in_shape.getDim(2), false, true);

  // Get Delta of w
  assert(in_shape.getDim(1) == out_shape.getDim(1));
  matrix::gemm(ins[0].getData(), ous[0].getGrad(), NULL, weight_.getGrad(),
      w_shape.getDim(1), in_shape.getDim(1), w_shape.getDim(2), true, false);

  // Get Delta of b
  matrix::matrix_to_vec(ous[0].getGrad(), weight_.getGrad(), out_shape.getDim(1),  out_shape.getDim(2), 1);
}


};
