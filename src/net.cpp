#include <vector>
#include "net.h"
#include "layer.h"


LogReg::LogReg(size_t inputDim, size_t hiddenDim):
  inputDim_(inputDim), hiddenDim_(hiddenDim) {
    layers_.emplace_back(FullyConnectedLayer(inputDim, hiddenDim));
    layers_.emplace_back(SigmoidLayer());
    layers_.emplace_back(NLLSigmoidLayer());
  }

SeqNet::forward(const std::vector<Tensor>& ins, std::vector<Tensor>& ous) {
  size_t n = layers_.size();
  if (1 == n) {
    layers_[0].forward(ins, ous);
  } else {
    std::vector<Tensor> tmps[n-1];
    layers_[0].forward(ins, tmps[0]);
    for (int i = 1; i < layers_.size() - 1; ++i) {
      layers_[i].forward(tmps[i-1], tmps[i]);
    layers_[n-1].forward(tmps[n-2], ous);
  }
}

SeqNet::backward(std::vector<Tensor> &ins) {
  for (int i = layers_.size() - 1; i >=0; --i) {

  }
}
