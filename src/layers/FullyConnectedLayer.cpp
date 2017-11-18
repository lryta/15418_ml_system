#include "layers/FullyConnectedLayer.h"
#include "MatrixOp.h"

namespace MLLib {

void FullyConnectedLayer::forward(const vector<tensor> &ins, vector<tensor> &ous) {
  assert(ins.size() == 1);
  auto in_shape = ins[0].getShape();
  auto w_shape = weight_.getShape();

  size_t data_num = in_shape.getDim(1);
  size_t matrx_row = in_shape.getDim(2);
  size_t matrx_col = in_shape.getDim(3);
  size_t w_col = w_shape.getDim(2);

  assert(in_shape.getDim(3) == weight_.getDim(1));
  shape out_shape(in_shape.getDim(1), in_shape.getDim(2), w_shape.getDim(2)) ;

  ous.emplace_back(out_shape);
  matrix::gemm(ins[0].get, weight_, bias_, ous[0],
      in_shape.getDim(2), in_shape.getDim(3), w_shape.getDim(3));
}

void FullyConnectedLayer::backward(vector<tensor> ins) {
  for (auto const*)
}

};
