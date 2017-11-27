#include <vector>
#include <assert>
#include "net.h"
#include "layer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/L2LossLayer.h"
#include "layers/negLogLikeliLayer.h"
#include "layers/ActivationLayer.h"

namespace TinyML {

using std::vector;

void MLPNet::buildLayers(shape in_shape, shape target_shape, vector<size_t> hiddenDims) {
  shape inter_shape, out_shape;
  inter_shape = input;

  for (auto const& hiddenDim : hiddenDims) {
    int batch_num = inter_shape.getDim(1);
    int inter_dim = inter_shape.getDim(2);

    layers_.push_back(new FullyConnectedLayer({inter_shape}, {out_shape}, hiddenDim));
    intermediate_tensors_.push_back(new Tensor(out_shape));

    assert(out_shape == shape(batch_num, hiddenDim));

    inter_shape = out_shape;
    layers_.push_back(new SigmoidLayer({inter_shape}, {out_shape}));
    intermediate_tensors_.push_back(new Tensor(out_shape));

    // shape shouldn't change after activation layer
    assert(inter_shape == out_shape);
  }

  // target shape should be same tointer_shape 
  assert(target_shape == inter_shape);
  layers_.push_back(new negLogLikeliLayer({inter_shape, target_shape}, {}));
}

vector<Tensor*> MLPNet::getParams() {
  if (params_.size() == 0) {
    for (auto const& layer : layers_) {
      vector<tensor*> layer_params = layers.getParams();
      std::insert(params_.end(), layer_params.begin(), layer_params.end());
    }
  }

  return params_;
}

MLPNet::~MLPNet() {
  for (auto const& ts : intermediate_tensors_)
    delete ts;

  for (auto const& layer : layers_)
    delete layer;
}

void MLPNet::forward(vector<Tensor*> ins, vector<Tensor*> targets) {
  assert(layers.size() > 1);

  for (int i = 0; i < layers_.size() - 1; ++i)
    layers_[i].forward({(i == 0)?(ins[0]):(intermediate_tensors_[i-1])}, {intermediate_tensors_[i]});

  layers_[layers_.size() - 1].forward({intermediate_tensors_[layers_.size()-2], targets[0]}, {});
}

int MLPNet::correctlyRecognizedDataNum() {
  return layers_[layers_.size() - 1].correctlyRecognizedDataNum();
}

float MLPNet::getLoss() {
  return layers_[layers_.size() - 1].getLoss();
}

void MLPNet::backward(vector<Tensor*> ins, vector<Tensor*> targets) {
  layers_[layers_.size() - 1].backward({ins[0], targets[0]}, {});

  for (int i = layers_.size() - 1; i >= 0; --i)
    layers_[i].backward({(i == 0)?(ins[0]):(intermediate_tensors_[i-1])}, {intermediate_tensors_[i]});
}

}
