#include <vector>
#include "net.h"
#include "layer.h"


LogReg::LogReg(int inputDim, int hiddenDim):
  inputDim_(inputDim), hiddenDim_(hiddenDim) {}

SeqNet::forward(const std::vector<Tensor>& ins, std::vector<Tensor>& ous) {
  size_t n = layers.size();
  if (1 == n) {
    layers[0].forward(ins, ous);
  } else {
    std::vector<Tensor> tmps[n-1];
    layers[0].forward(ins, tmps[0]);
    for (int i = 1; i < layers.size() - 1; ++i) {
      layers[i].forward(tmps[i-1], tmps[i]);
    layers[n-1].forward(tmps[n-2], ous);
  }
}

SeqNet::backward
