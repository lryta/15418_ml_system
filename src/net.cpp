#include <vector>
#include <cassert>
#include "net.h"
#include "layer.h"
#include "layers/fullyConnectedLayer.h"
#include "layers/l2LossLayer.h"
#include "layers/negLogLikeliLayer.h"
#include "layers/activationLayer.h"

namespace TinyML {

using std::vector;

void MLPnet::buildlayers(shape in_shape, shape target_shape, vector<size_t> hiddenDims) {
  shape inter_shape = in_shape, out_shape(-1);

  vector<shape> in_shapes;
  vector<shape> out_shapes;

  for (auto const& hiddenDim : hiddenDims) {
    int batch_num = inter_shape.getDim(1);
    int inter_dim = inter_shape.getDim(2);

    in_shapes.clear();
    in_shapes.push_back(inter_shape);
    out_shapes.clear();
    layers_.push_back(new FullyConnectedlayer(in_shapes, out_shapes, hiddenDim));
    intermediate_tensors_.push_back(new tensor(out_shape));

    assert(out_shapes[0] == shape(batch_num, hiddenDim));

    inter_shape = out_shapes[0];

    in_shapes.clear();
    in_shapes.push_back(inter_shape);
    out_shapes.clear();
    layers_.push_back(new Sigmoidlayer(in_shapes, out_shapes));
    intermediate_tensors_.push_back(new tensor(out_shape));

    // shape shouldn't change after activation layer
    assert(in_shapes[0] == out_shapes[0]);
  }

  // target shape should be same tointer_shape 
  assert(target_shape == in_shapes[0]);
  out_shapes.clear();
  in_shapes.push_back(target_shape);
  layers_.push_back(new negLogLikelilayer(in_shapes, out_shapes));
}

vector<tensor*> MLPnet::getParams() {
  if (params_.size() == 0) {
    for (auto const& layer : layers_) {
      vector<tensor*> layer_params = layer->getParam();
      params_.insert(params_.end(), layer_params.begin(), layer_params.end());
    }
  }

  return params_;
}

MLPnet::~MLPnet() {
  for (auto const& ts : intermediate_tensors_)
    delete ts;

  for (auto const& layer : layers_)
    delete layer;
}

void MLPnet::forward(vector<tensor*> ins, vector<tensor*> targets) {
  assert(layers_.size() >= 2);

  for (int i = 0; i <= (int)layers_.size() - 2; ++i)
    layers_[i]->forward({(i == 0)?(ins[0]):(intermediate_tensors_[i-1])}, {intermediate_tensors_[i]});

  layers_.back()->forward({intermediate_tensors_[layers_.size()-2], targets[0]}, {});
}

int MLPnet::correctlyRecognizedDataNum() {
  return dynamic_cast<Losslayer*>(layers_.back())->correctlyRecognizedDataNum();
}

float MLPnet::getLoss() {
  return dynamic_cast<Losslayer*>(layers_.back())->getLoss();
}

void MLPnet::backward(vector<tensor*> ins, vector<tensor*> targets) {
  layers_.back()->backward({ins[0], targets[0]}, {});

  for (int i = (int)layers_.size() - 2; i >= 0; --i)
    layers_[i]->backward({(i == 0)?(ins[0]):(intermediate_tensors_[i-1])}, {intermediate_tensors_[i]});
}

}
