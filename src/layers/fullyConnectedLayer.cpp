#include "layers/fullyConnectedLayer.h"
#include "operations/matrixOp.h"

namespace TinyML {

vector<tensor*> FullyConnectedlayer::getParam() {
  return {&weight_, &bias_};
}

void FullyConnectedlayer::inferShape(vector<shape> &ins, vector<shape> &ous) {
  auto in_shape = ins[0];
  ous.clear();
  ous.push_back(shape(in_shape.getDim(1), inter_dim_));
}

void FullyConnectedlayer::initWeight(vector<shape> &ins, vector<shape> &ous) {
  weight_ = tensor(shape(ins[0].getDim(2), inter_dim_));
  bias_ = tensor(shape(inter_dim_));
}

void FullyConnectedlayer::forward(vector<tensor*> ins, vector<tensor*> ous) {
  assert(ins.size() == 1);
  assert(ous.size() == 1);
  auto in_shape = ins[0]->getShape();
  auto out_shape = ous[0]->getShape();
  auto bias_shape = bias_.getShape();
  auto w_shape = weight_.getShape();

  assert(in_shape.getDim(1) == out_shape.getDim(1));
  assert(w_shape.getDim(2) == out_shape.getDim(2));
  assert(in_shape.getDim(2) == w_shape.getDim(1));
  assert(bias_shape.getDim(1) == w_shape.getDim(2));

#ifdef COMPILE_CUDA
  ins[0]->SyncDataCPUToGPU();
  weight_.SyncDataCPUToGPU();
  bias_.SyncDataCPUToGPU();
  // y = x * w + b
  matrix::gemm(ins[0]->getGPUData(), weight_.getGPUData(), bias_.getGPUData(), ous[0]->getGPUData(),
      out_shape.getDim(1), in_shape.getDim(2), out_shape.getDim(2));

  ous[0]->SyncDataGPUToCPU();
#else
  // y = x * w + b
  matrix::gemm(ins[0]->getData(), weight_.getData(), bias_.getData(), ous[0]->getData(),
      out_shape.getDim(1), in_shape.getDim(2), out_shape.getDim(2));
#endif
}

void FullyConnectedlayer::backward(vector<tensor*> ins, vector<tensor*> ous) {
  assert(ins.size() > 0);
  assert(ous.size() > 0);
  auto in_shape = ins[0]->getShape();
  auto w_shape = weight_.getShape();
  auto out_shape = ous[0]->getShape();

#ifdef COMPILE_CUDA
  // ins & Weight GPU data is already synced in forward
  ous[0]->SyncGradCPUToGPU();
  
  // Get Delta of ins
  // x_grad = y_grad * trans(w)
  assert(out_shape.getDim(2) == w_shape.getDim(2));
  matrix::gemm(ous[0]->getGPUGrad(), weight_.getGPUData(), NULL, ins[0]->getGPUGrad(),
      in_shape.getDim(1), out_shape.getDim(2), in_shape.getDim(2), false, true);

  // Get Delta of w
  // w_grad = trans(x) * y_grad
  assert(in_shape.getDim(1) == out_shape.getDim(1));
  matrix::gemm(ins[0]->getGPUData(), ous[0]->getGPUGrad(), NULL, weight_.getGPUGrad(),
      w_shape.getDim(1), in_shape.getDim(1), w_shape.getDim(2), true, false);

  ins[0]->SyncGradGPUToCPU();
  weight_.SyncGradGPUToCPU();
#else
  // Get Delta of ins
  // x_grad = y_grad * trans(w)
  assert(out_shape.getDim(2) == w_shape.getDim(2));
  matrix::gemm(ous[0]->getGrad(), weight_.getData(), NULL, ins[0]->getGrad(),
      in_shape.getDim(1), out_shape.getDim(2), in_shape.getDim(2), false, true);

  // Get Delta of w
  // w_grad = trans(x) * y_grad
  assert(in_shape.getDim(1) == out_shape.getDim(1));
  matrix::gemm(ins[0]->getData(), ous[0]->getGrad(), NULL, weight_.getGrad(),
      w_shape.getDim(1), in_shape.getDim(1), w_shape.getDim(2), true, false);
#endif

  // Get Delta of b
  // bias_grad = sum(y_grad, 1)
  matrix::reduceMatrix(ous[0]->getGrad(), bias_.getGrad(), out_shape.getDim(1),  out_shape.getDim(2), 1);
}

};
