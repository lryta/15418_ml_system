#include <vector>
#include "net.h"
#include "layer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/L2LossLayer.h"
#include "layers/ActivationLayer.h"

namespace MLLib {

using std::vector;

void MLPNet::buildLayers(shape input, vector<size_t> hiddenDims) {
  shape in_shape, out_shape;
  in_shape = input;

  for (auto const& hiddenDim : hiddenDim) {
    layers_.push_back(new FullyConnectedLayer(in_shape, out_shape, hiddenDim));
    intermediate_tensors_.push_back(new Tensor(out_shape));

    in_shape = out_shape;
    layers_.push_back(new SigmoidLayer(in_shape, out_shape));
    intermediate_tensors_.push_back(new Tensor(out_shape));

    in_shape = out_shape;
  }

  // TODO(Larry): replace l2 to nll-sigmoid layer
  // TODO(Larry): Define nll-sigmoid layer
  auto target_shape = in_shape;
  layers_.push_back(new L2LossLayer({in_shape, target_shape}, {}));
}

vector<Tensor*> MLPNet::getParams() {
  if (params_.size() == 0) {
    for (auto const& layer : layers_) {
      auto layer_param = layers.getParams();
      std::insert(param_.end(), layer_param.begin(), layer_param.end());
    }
  }

  return params_;
}

MLPNet::~MLPNet() {
  for (auto const& ts : intermediate_tensors_)
    delete ts;

  for (auto const& layer : intermediate_tensors_)
    delete layer;
}

void MLPNet::forward(vector<Tensor*> ins, vector<Tensor*> targets) {
  assert(layers.size() > 1);

  for (int i = 0; i < layers_.size() - 1; ++i)
    layers_[i].forward((i == 0)?(ins[0]):intermediate_tensors_[i-1], intermediate_tensors_[i]);

  layers_[layers_.size() - 1].forward({intermediate_tensors_[layers_.size()-2], targets[0]}, {});
}

void MLPNet::backward(vector<Tensor*> ins, vector<Tensor*> targets) {
  layers_[layers_.size() - 1].backword({ins[0], targets[0]}, {});

  for (int i = layers_.size() - 1; i >= 0; --i)
    layers_[i].backward((i == 0)?(ins[0]):intermediate_tensors_[i-1], intermediate_tensors_[i]);
}

}
